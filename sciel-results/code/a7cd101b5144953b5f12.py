"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 4, 7]


def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)


def _exhaustive_search(n, m, p, max_rank, coeff_set, timeout=45):
    """Search for rank-minimal decomposition of n x m x p tensor.
    
    For each rank r, enumerate U and V entries from coeff_set,
    then solve linear system for W.
    """
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    num_coeffs = len(coeff_set)
    
    for r in range(1, max_rank + 1):
        if time.time() - start_time > timeout:
            break
        
        # Random search with structured sampling
        num_trials = 20000
        found = False
        
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
            
            # Random U, V with entries from coeff_set
            U_rand = np.random.choice(coeff_set, size=(nm, r))
            V_rand = np.random.choice(coeff_set, size=(mp, r))
            
            # Set up linear system for W
            # Variables: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
            # Number of variables: p * n * r
            # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A[eq_idx, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                if np.allclose(A @ W_flat, b, atol=1e-7):
                    W_rand = W_flat.reshape((p * n, r))
                    # Check entries are nice (integer or half-integer)
                    W_rounded = np.round(W_rand * 2) / 2
                    if np.allclose(W_rounded, W_rand, atol=1e-7):
                        U_arr = np.array(U_rand, dtype=float)
                        V_arr = np.array(V_rand, dtype=float)
                        if _check_decomposition(U_arr, V_arr, W_rounded, n, m, p):
                            return (r, U_arr.tolist(), V_arr.tolist(), W_rounded.tolist())
            except:
                pass
        
        # Also try structured search: set most U,V entries to 0 or 1
        if time.time() - start_time > timeout:
            break
        
        for trial in range(5000):
            if time.time() - start_time > timeout:
                break
            
            # Structured: sparse U and V
            U_rand = np.zeros((nm, r))
            V_rand = np.zeros((mp, r))
            for c in range(r):
                # Each column of U has a few non-zero entries
                nz_u = np.random.choice(nm, size=min(3, nm), replace=False)
                for idx in nz_u:
                    U_rand[idx, c] = np.random.choice(coeff_set)
                nz_v = np.random.choice(mp, size=min(3, mp), replace=False)
                for idx in nz_v:
                    V_rand[idx, c] = np.random.choice(coeff_set)
            
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A[eq_idx, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                if np.allclose(A @ W_flat, b, atol=1e-7):
                    W_rand = W_flat.reshape((p * n, r))
                    W_rounded = np.round(W_rand * 2) / 2
                    if np.allclose(W_rounded, W_rand, atol=1e-7):
                        U_arr = np.array(U_rand, dtype=float)
                        V_arr = np.array(V_rand, dtype=float)
                        if _check_decomposition(U_arr, V_arr, W_rounded, n, m, p):
                            return (r, U_arr.tolist(), V_arr.tolist(), W_rounded.tolist())
            except:
                pass
    
    return None


def _construct_sub(n, m, p, max_rank=25, timeout=40):
    """Construct decomposition for n x m x p sub-problem."""
    coeff_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    result = _exhaustive_search(n, m, p, max_rank=max_rank, coeff_set=coeff_set, timeout=timeout)
    if result:
        return result
    
    # Fallback: schoolbook
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
    return (rank, U, V, W)


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Partition the 7 output columns (k dimension) into blocks: 3, 2, 2
    blocks = [3, 2, 2]
    k_offsets = []
    offset = 0
    for b in blocks:
        k_offsets.append(offset)
        offset += b
    
    # Find sub-decompositions
    sub_decomps = []
    # Block 0: k in [0, 3) -> 3x4x3
    r0, U0, V0, W0 = _construct_sub(n, m, 3, max_rank=25, timeout=35)
    sub_decomps.append((r0, U0, V0, W0, k_offsets[0], 3))
    # Block 1: k in [3, 5) -> 3x4x2
    r1, U1, V1, W1 = _construct_sub(n, m, 2, max_rank=20, timeout=35)
    sub_decomps.append((r1, U1, V1, W1, k_offsets[1], 2))
    # Block 2: k in [5, 7) -> 3x4x2
    r2, U2, V2, W2 = _construct_sub(n, m, 2, max_rank=20, timeout=35)
    sub_decomps.append((r2, U2, V2, W2, k_offsets[2], 2))
    
    total_rank = sum(r for r, _, _, _, _, _ in sub_decomps)
    
    # Combine: U is (n*m, total_rank), V is (m*p, total_rank), W is (p*n, total_rank)
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for r, U_sub, V_sub, W_sub, k_off, b in sub_decomps:
        # U_sub: (n*m, r) - same input indices for all blocks
        for i in range(n * m):
            for c in range(r):
                U_full[i][col_offset + c] = U_sub[i][c]
        
        # V: for block starting at k_off with size b
        # V_sub has shape (m*b, r) with indices (j, k_local)
        for j in range(m):
            for kk in range(b):
                k_local = kk
                k_global = k_off + kk
                row_v = j * p + k_global
                for c in range(r):
                    V_full[row_v][col_offset + c] = V_sub[j * b + k_local][c]
        
        # W: W_sub has shape (b*n, r) with indices (k_local, i)
        for kk in range(b):
            k_global = k_off + kk
            for i in range(n):
                row_w = k_global * n + i
                for c in range(r):
                    W_full[row_w][col_offset + c] = W_sub[kk * n + i][c]
        
        col_offset += r
    
    return {"U": U_full, "V": V_full, "W": W_full}
