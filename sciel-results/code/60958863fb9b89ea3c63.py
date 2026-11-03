"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build Strassen-based decomposition
    result = _build_strassen_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            # Try to reduce rank by zeroing redundant columns
            U, V, W, rank = _try_reduce_rank(U, V, W, rank, n, m, p, deadline)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _try_reduce_rank(U, V, W, rank, n, m, p, deadline):
    """
    Try to reduce rank by identifying and removing linearly dependent rank terms.
    For each column c, check if removing it still allows the decomposition to work
    (i.e., the missing tensor components can be compensated by other columns).
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Build the tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # We want T[i,j,k] = 1 for all i,j,k
    
    # Try removing columns one by one (greedy)
    # For each column, check if the residual can be absorbed
    # This is expensive, so do it selectively
    
    # Instead, try a simpler approach: check for duplicate or proportional columns
    # across U, V, W
    
    current_rank = rank
    removed = set()
    
    # Check for columns that are scalar multiples of others in all three matrices
    # This is O(r^2 * (nm + mp + pn)) which is manageable for r=84
    
    for c1 in range(current_rank):
        if c1 in removed:
            continue
        if time.time() > deadline - 5:
            break
        for c2 in range(c1 + 1, current_rank):
            if c2 in removed:
                continue
            # Check if column c2 is a scalar multiple of c1 in all three matrices
            u1 = U[:, c1]
            u2 = U[:, c2]
            v1 = V[:, c1]
            v2 = V[:, c2]
            w1 = W[:, c1]
            w2 = W[:, c2]
            
            # Find scalar from U
            nonzero_u1 = np.nonzero(u1)[0]
            if len(nonzero_u1) == 0:
                continue
            idx = nonzero_u1[0]
            if u1[idx] == 0:
                continue
            scale = u2[idx] / u1[idx]
            
            # Check if scale works for all
            if (np.allclose(u1 * scale, u2, atol=1e-9) and
                np.allclose(v1 * scale, v2, atol=1e-9) and
                np.allclose(w1 * scale, w2, atol=1e-9)):
                # c2 is redundant, remove it
                removed.add(c2)
    
    if removed:
        keep = [c for c in range(rank) if c not in removed]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        current_rank = len(keep)
    
    return U, V, W, current_rank


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4, 7) indexed as U[i*2+j, c] for A[i,j]
    V: (4, 7) indexed as V[j*2+k, c] for B[j,k]
    W: (4, 7) indexed as W[k*2+i, c] for C[k,i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6
    
    # V: B factors
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6
    
    # W: C factors (W[k*2+i, c])
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    W[2, 2] = 1   # m2 in C[0,1]
    W[2, 4] = 1   # m4 in C[0,1]
    W[1, 1] = 1   # m1 in C[1,0]
    W[1, 3] = 1   # m3 in C[1,0]
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    
    Per k-block:
      Top 2 rows (i=0,1):
        j in {0,1}: Strassen 2x2x2 (rank 7)
        j in {2,3}: Strassen 2x2x2 (rank 7)
        j = 4: schoolbook 2x1x2 (rank 4)
      Bottom row (i=2):
        j in {0,1}: schoolbook 1x2x2 (rank 4)
        j in {2,3}: schoolbook 1x2x2 (rank 4)
        j = 4: schoolbook 1x1x2 (rank 2)
    Total per k-block: 7+7+4+4+4+2 = 28
    3 k-blocks: 84
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2  # start k index
        k1 = k0 + 1       # end k index
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]   # i=0, j=0
            U[0*5 + 1, col] = sU[1, c]   # i=0, j=1
            U[1*5 + 0, col] = sU[2, c]   # i=1, j=0
            U[1*5 + 1, col] = sU[3, c]   # i=1, j=1
            V[0*6 + k0, col] = sV[0, c]   # j=0, k=k0
            V[0*6 + k1, col] = sV[1, c]   # j=0, k=k1
            V[1*6 + k0, col] = sV[2, c]   # j=1, k=k0
            V[1*6 + k1, col] = sV[3, c]   # j=1, k=k1
            W[k0*3 + 0, col] = sW[0, c]   # k=k0, i=0
            W[k0*3 + 1, col] = sW[1, c]   # k=k0, i=1
            W[k1*3 + 0, col] = sW[2, c]   # k=k1, i=0
            W[k1*3 + 1, col] = sW[3, c]   # k=k1, i=1
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    
    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
