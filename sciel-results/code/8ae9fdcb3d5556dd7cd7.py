"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Primary approach: Kronecker product of Strassen (2x2x2, rank 7) and schoolbook (1x2x3, rank 6)
gives rank 42. Additionally, attempt direct low-rank search via tensor flattening.

The multiplication tensor T[i,j,k] = 1 for all i in 0..1, j in 0..3, k in 0..5
must be decomposed as sum_s U[i*4+j, s] * V[j*6+k, s] * W[k*2+i, s].
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]


def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7)."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1

    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1

    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1

    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1

    # M5 = (A00+A01)B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    return U1, V1, W1


def _rect_1x2x3_factors():
    """Return (U2, V2, W2) for 1x2x3 schoolbook core, shapes (2,6),(6,6),(3,6)."""
    U2 = np.zeros((2, 6), dtype=float)
    V2 = np.zeros((6, 6), dtype=float)
    W2 = np.zeros((3, 6), dtype=float)

    for j in range(2):
        for k in range(3):
            s = j * 3 + k
            U2[j, s] = 1
            V2[j * 3 + k, s] = 1
            W2[k, s] = 1

    return U2, V2, W2


def _build_rank42_kron():
    """Build rank-42 decomposition via Kronecker product of Strassen (2x2x2) and schoolbook (1x2x3)."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    U2, V2, W2 = _rect_1x2x3_factors()

    r1, r2 = 7, 6
    total_rank = r1 * r2

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for s1 in range(r1):
        for s2 in range(r2):
            s = s1 * r2 + s2

            for i1 in range(2):
                for j1 in range(2):
                    u1_val = U1[i1 * 2 + j1, s1]
                    if u1_val == 0:
                        continue
                    for i2 in range(1):
                        for j2 in range(2):
                            u2_val = U2[i2 * 2 + j2, s2]
                            if u2_val == 0:
                                continue
                            row_u = (i1 * 1 + i2) * 4 + (j1 * 2 + j2)
                            U[row_u, s] += u1_val * u2_val

            for j1 in range(2):
                for k1 in range(2):
                    v1_val = V1[j1 * 2 + k1, s1]
                    if v1_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v2_val = V2[j2 * 3 + k2, s2]
                            if v2_val == 0:
                                continue
                            row_v = (j1 * 2 + j2) * 6 + (k1 * 3 + k2)
                            V[row_v, s] += v1_val * v2_val

            for k1 in range(2):
                for i1 in range(2):
                    w1_val = W1[k1 * 2 + i1, s1]
                    if w1_val == 0:
                        continue
                    for k2 in range(3):
                        for i2 in range(1):
                            w2_val = W2[k2 * 1 + i2, s2]
                            if w2_val == 0:
                                continue
                            row_w = (k1 * 3 + k2) * 2 + (i1 * 1 + i2)
                            W[row_w, s] += w1_val * w2_val

    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s] for s in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_direct_low_rank(n, m, p, target_rank, time_budget=30.0, seed=42):
    """Attempt to find a rank-`target_rank` decomposition via iterative refinement.
    
    Uses the tensor flattening approach: for each (i,j,k), we need
    sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
    
    Strategy: initialize with the rank-42 solution projected to target_rank,
    then iterate: fix two factors, solve for the third via least squares.
    """
    start = time.time()
    rng = np.random.RandomState(seed)
    
    # Start from rank-42 solution
    U42, V42, W42 = _build_rank42_kron()
    
    # Project to target_rank by selecting columns with largest norms
    # Or use SVD-based compression
    def _compress_factors(U, V, W, target_r):
        """Compress rank-r factors to rank-target_r via SVD of the tensor."""
        # Build the full tensor T of shape (n*m, m*p, p*n) flattened
        # T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
        # For compression, we use the mode-1, mode-2, mode-3 SVDs
        
        # Mode-1 unfolding: T1[a, (b,c)] = sum_s U[a,s]*V[b,s]*W[c,s]
        # This is U @ diag(V[:,s]*W[c,s])... actually it's more complex.
        
        # Simpler: just randomly select target_r columns from the rank-42 solution
        # weighted by column norm
        col_norms_U = np.linalg.norm(U, axis=0)
        col_norms_V = np.linalg.norm(V, axis=0)
        col_norms_W = np.linalg.norm(W, axis=0)
        
        # Select top target_r columns by combined norm
        combined = col_norms_U * col_norms_V * col_norms_W
        top_cols = np.argsort(combined)[-target_r:]
        top_cols = np.sort(top_cols)
        
        U_c = U[:, top_cols]
        V_c = V[:, top_cols]
        W_c = W[:, top_cols]
        return U_c, V_c, W_c
    
    U_c, V_c, W_c = _compress_factors(U42, V42, W42, target_rank)
    
    # Iterative refinement: alternate fixing two factors and solving for the third
    # For fixed V and W, solve for U: 
    #   For each (i,j): sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1 for all k
    #   This is a linear system in U[i*m+j,:] for each (i,j)
    #   Actually, for fixed (i,j), we need: for all k: sum_s U[i*m+j,s] * (V[j*p+k,s] * W[k*n+i,s]) = 1
    #   Let M_{k,s} = V[j*p+k,s] * W[k*n+i,s], then sum_s U[i*m+j,s] * M_{k,s} = 1 for all k
    #   This is M @ U[i*m+j,:] = ones(p)
    
    best_err = np.inf
    best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
    
    for iteration in range(200):
        if time.time() - start > time_budget:
            break
        
        # Fix V and W, solve for U
        U_new = np.zeros((n * m, target_rank), dtype=float)
        for i in range(n):
            for j in range(m):
                # M[k, s] = V[j*p+k, s] * W[k*n+i, s]
                M = np.zeros((p, target_rank), dtype=float)
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M @ u = ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except:
                    U_new[i * m + j, :] = U_c[i * m + j, :]
        
        # Fix U and W, solve for V
        V_new = np.zeros((m * p, target_rank), dtype=float)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank), dtype=float)
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except:
                    V_new[j * p + k, :] = V_c[j * p + k, :]
        
        # Fix U and V, solve for W
        W_new = np.zeros((p * n, target_rank), dtype=float)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank), dtype=float)
                for j in range(m):
                    M[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except:
                    W_new[k * n + i, :] = W_c[k * n + i, :]
        
        U_c, V_c, W_c = U_new, V_new, W_new
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_c[i * m + j, s] * V_c[j * p + k, s] * W_c[k * n + i, s] for s in range(target_rank))
                    err = max(err, abs(val - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-9:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err


def _to_output(mat):
    """Convert numpy array to list of lists with int where possible."""
    return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj][col] = 1
                V[jj * p + kk][col] = 1
                W[kk * n + ii][col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    # Primary: rank-42 Kronecker product (Strassen ⊗ 1x2x3 schoolbook)
    U42, V42, W42 = _build_rank42_kron()

    if _verify(U42, V42, W42, n, m, p):
        # Try to find a lower-rank decomposition using remaining time
        remaining = deadline - time.time()
        
        if remaining > 20:
            # Try rank 40
            time_per_attempt = min(remaining / 3, 35.0)
            for target_r in [40, 41]:
                if time.time() > deadline - 15:
                    break
                U_try, V_try, W_try, err = _try_direct_low_rank(n, m, p, target_r, time_budget=time_per_attempt)
                if U_try is not None and err < 1e-9:
                    return {"U": _to_output(U_try), "V": _to_output(V_try), "W": _to_output(W_try)}
        
        return {"U": _to_output(U42), "V": _to_output(V42), "W": _to_output(W42)}

    # Fallback: schoolbook rank 48
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}
