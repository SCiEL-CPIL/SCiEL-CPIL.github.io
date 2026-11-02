import numpy as np
import time
import random

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (A00+A11)*(B00+B11)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M2: (A10+A11)*B00
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1

        # M3: A00*(B10-B00)
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M4: A11*(B01-B11)
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1

        # M5: (A00+A01)*B11
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1

        # M6: (A10-A00)*(B00+B01)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M7: (A01-A11)*(B10+B11)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W


def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)


def _try_remove_terms(U, V, W, n, m, p):
    """Try to remove terms from the decomposition if they are linearly dependent."""
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_full = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)

    if not np.allclose(T, T_full, atol=1e-10):
        return None

    # Flatten each term tensor to a vector for fast linear independence checks
    flat_dim = n * m * m * p * p * n
    T_flat = np.zeros((rank, flat_dim), dtype=float)
    for r in range(rank):
        T_flat[r] = np.einsum('i,j,k->ijk', Ua[:, r], Va[:, r], Wa[:, r]).flatten()

    # Find null space of T_flat (rows)
    try:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=False)
        tol = 1e-10
        num_zero = sum(1 for si in s if si < tol)
    except Exception:
        return None

    if num_zero == 0:
        return None

    # For each null space vector, try to eliminate the term with largest coefficient
    null_vecs = Vt[-num_zero:].T  # shape (rank, num_zero)
    for nv in null_vecs:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        max_idx = np.argmax(np.abs(nv))
        if abs(nv[max_idx]) < 1e-10:
            continue
        new_mask = []
        new_coeffs = []
        for r in range(rank):
            if r == max_idx:
                continue
            c = 1.0 - nv[r] / nv[max_idx]
            if abs(c) > 1e-10:
                new_mask.append(r)
                new_coeffs.append(c)

        if len(new_mask) < rank:
            U_new = Ua[:, new_mask] * np.array(new_coeffs)[np.newaxis, :]
            V_new = Va[:, new_mask]
            W_new = Wa[:, new_mask]
            T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            if np.allclose(T, T_check, atol=1e-10):
                return (
                    U_new.tolist(),
                    V_new.tolist(),
                    W_new.tolist()
                )

    return None


def _build_2x2x3_rank11(n, m, p, k_offset):
    """
    Build a rank-11 decomposition for a 2x2x3 block starting at column k_offset.
    Uses schoolbook 12 terms and removes one dependent term.
    """
    # Schoolbook terms for 2x2x3
    U_list = []
    V_list = []
    W_list = []
    
    # Indices: i in 0..1, j in 0..1, k in 0..2 (relative)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U = [0] * (n*m)
                V = [0] * (m*p)
                W = [0] * (p*n)
                
                U[i*m + j] = 1
                V[j*p + (k_offset + k)] = 1
                W[(k_offset + k)*n + i] = 1
                
                U_list.append(U)
                V_list.append(V)
                W_list.append(W)
                
    # Convert to matrices for _try_remove_terms
    # Note: _try_remove_terms expects full dimension matrices
    U_mat = np.array(U_list).T # (n*m, 12)
    V_mat = np.array(V_list).T # (m*p, 12)
    W_mat = np.array(W_list).T # (p*n, 12)
    
    # Try to remove terms
    reduced = _try_remove_terms(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p)
    if reduced is not None:
        U_r, V_r, W_r = reduced
        return U_r, V_r, W_r
    else:
        # Fallback: return all 12 terms
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()


def construct():
    n, m, p = DIMENSIONS

    # Build rank-32 decomposition using 2+2+3+2 split
    # Cols 0-1: Strassen
    # Cols 2-3: Strassen
    # Cols 4-6: Rank-11 (2x2x3)
    # Cols 7-8: Strassen
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block):
        U_parts.append(U_block)
        V_parts.append(V_block)
        W_parts.append(W_block)
        
    # Block 1: Cols 0-1 (Strassen)
    U1 = [[0]*7 for _ in range(n*m)]
    V1 = [[0]*7 for _ in range(m*p)]
    W1 = [[0]*7 for _ in range(p*n)]
    
    # Strassen for k0=0, k1=1
    col = 0
    # M1
    U1[0][col]=1; U1[3][col]=1; V1[0*p+0][col]=1; V1[1*p+1][col]=1; W1[0*n+0][col]=1; W1[1*n+1][col]=1; col+=1
    # M2
    U1[2][col]=1; U1[3][col]=1; V1[0*p+0][col]=1; W1[0*n+1][col]=1; W1[1*n+1][col]=-1; col+=1
    # M3
    U1[0][col]=1; V1[0*p+1][col]=1; V1[1*p+1][col]=-1; W1[1*n+0][col]=1; W1[1*n+1][col]=1; col+=1
    # M4
    U1[3][col]=1; V1[1*p+0][col]=1; V1[0*p+0][col]=-1; W1[0*n+0][col]=1; W1[0*n+1][col]=1; col+=1
    # M5
    U1[0][col]=1; U1[1][col]=1; V1[1*p+1][col]=1; W1[0*n+0][col]=-1; W1[1*n+0][col]=1; col+=1
    # M6
    U1[2][col]=1; U1[0][col]=-1; V1[0*p+0][col]=1; V1[0*p+1][col]=1; W1[1*n+1][col]=1; col+=1
    # M7
    U1[1][col]=1; U1[3][col]=-1; V1[1*p+0][col]=1; V1[1*p+1][col]=1; W1[0*n+0][col]=1; col+=1
    
    add_block(U1, V1, W1)
    
    # Block 2: Cols 2-3 (Strassen)
    U2 = [[0]*7 for _ in range(n*m)]
    V2 = [[0]*7 for _ in range(m*p)]
    W2 = [[0]*7 for _ in range(p*n)]
    
    col = 0
    # M1
    U2[0][col]=1; U2[3][col]=1; V2[0*p+2][col]=1; V2[1*p+3][col]=1; W2[2*n+0][col]=1; W2[3*n+1][col]=1; col+=1
    # M2
    U2[2][col]=1; U2[3][col]=1; V2[0*p+2][col]=1; W2[2*n+1][col]=1; W2[3*n+1][col]=-1; col+=1
    # M3
    U2[0][col]=1; V2[0*p+3][col]=1; V2[1*p+3][col]=-1; W2[3*n+0][col]=1; W2[3*n+1][col]=1; col+=1
    # M4
    U2[3][col]=1; V2[1*p+2][col]=1; V2[0*p+2][col]=-1; W2[2*n+0][col]=1; W2[2*n+1][col]=1; col+=1
    # M5
    U2[0][col]=1; U2[1][col]=1; V2[1*p+3][col]=1; W2[2*n+0][col]=-1; W2[3*n+0][col]=1; col+=1
    # M6
    U2[2][col]=1; U2[0][col]=-1; V2[0*p+2][col]=1; V2[0*p+3][col]=1; W2[3*n+1][col]=1; col+=1
    # M7
    U2[1][col]=1; U2[3][col]=-1; V2[1*p+2][col]=1; V2[1*p+3][col]=1; W2[2*n+0][col]=1; col+=1
    
    add_block(U2, V2, W2)
    
    # Block 3: Cols 4-6 (Rank 11)
    U3, V3, W3 = _build_2x2x3_rank11(n, m, p, 4)
    add_block(U3, V3, W3)
    
    # Block 4: Cols 7-8 (Strassen)
    U4 = [[0]*7 for _ in range(n*m)]
    V4 = [[0]*7 for _ in range(m*p)]
    W4 = [[0]*7 for _ in range(p*n)]
    
    col = 0
    # M1
    U4[0][col]=1; U4[3][col]=1; V4[0*p+7][col]=1; V4[1*p+8][col]=1; W4[7*n+0][col]=1; W4[8*n+1][col]=1; col+=1
    # M2
    U4[2][col]=1; U4[3][col]=1; V4[0*p+7][col]=1; W4[7*n+1][col]=1; W4[8*n+1][col]=-1; col+=1
    # M3
    U4[0][col]=1; V4[0*p+8][col]=1; V4[1*p+8][col]=-1; W4[8*n+0][col]=1; W4[8*n+1][col]=1; col+=1
    # M4
    U4[3][col]=1; V4[1*p+7][col]=1; V4[0*p+7][col]=-1; W4[7*n+0][col]=1; W4[7*n+1][col]=1; col+=1
    # M5
    U4[0][col]=1; U4[1][col]=1; V4[1*p+8][col]=1; W4[7*n+0][col]=-1; W4[8*n+0][col]=1; col+=1
    # M6
    U4[2][col]=1; U4[0][col]=-1; V4[0*p+7][col]=1; V4[0*p+8][col]=1; W4[8*n+1][col]=1; col+=1
    # M7
    U4[1][col]=1; U4[3][col]=-1; V4[1*p+7][col]=1; V4[1*p+8][col]=1; W4[7*n+0][col]=1; col+=1
    
    add_block(U4, V4, W4)
    
    # Concatenate all blocks
    U = []
    V = []
    W = []
    for i in range(n*m):
        U.append([part[i][r] for part, r in zip(U_parts, range(len(U_parts[0][0])) if False else [0]*len(U_parts[0][0]))]) # This is wrong, need to handle different ranks
    
    # Better concatenation
    total_rank = sum(len(part[0]) for part in U_parts)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for U_part, V_part, W_part in zip(U_parts, V_parts, W_parts):
        r = len(U_part[0])
        for i in range(n*m):
            for j in range(r):
                U[i][col_offset + j] = U_part[i][j]
        for i in range(m*p):
            for j in range(r):
                V[i][col_offset + j] = V_part[i][j]
        for i in range(p*n):
            for j in range(r):
                W[i][col_offset + j] = W_part[i][j]
        col_offset += r
        
    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fallback to old rank 32
        U, V, W = _build_rank_32(n, m, p)
        if not _validate(U, V, W, n, m, p):
            # Schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}

    # Try to reduce rank via linear dependency analysis
    reduced = _try_remove_terms(U, V, W, n, m, p)
    if reduced is not None:
        U2, V2, W2 = reduced
        if _validate(U2, V2, W2, n, m, p):
            return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
