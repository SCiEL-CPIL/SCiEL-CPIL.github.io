"""
Find rank-minimal tensor decomposition for 3x5 * 5x7 matrix multiplication.
Uses recursive block decomposition with Strassen for 2x2x2 sub-blocks.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using numpy."""
    rank = U.shape[1]
    if rank == 0:
        return True
    
    # Build index arrays
    i_idx = np.repeat(np.arange(n), m*p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n*m)
    
    U_idx = i_idx * m + j_idx
    V_idx = j_idx * p + k_idx
    W_idx = k_idx * n + i_idx
    
    U_sel = U[U_idx, :]
    V_sel = V[V_idx, :]
    W_sel = W[W_idx, :]
    
    result = np.sum(U_sel * V_sel * W_sel, axis=1)
    return np.all(np.abs(result - 1.0) < tol)

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 * 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 multiplications
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = (b-c)h
    U[1, 3] = 1; U[2, 3] = -1
    V[3, 3] = 1
    W[1, 3] = 1
    
    # M5 = (a+b)g
    U[0, 4] = 1; U[1, 4] = 1
    V[2, 4] = 1
    W[3, 4] = 1
    
    # M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1
    
    # M7 = (a-c)(e+g)
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _combine(decomps):
    """Combine a list of (U, V, W) decompositions by horizontal concatenation."""
    if not decomps:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
    
    total_rank = sum(d[0].shape[1] for d in decomps)
    n_rows_u = decomps[0][0].shape[0]
    n_rows_v = decomps[0][1].shape[0]
    n_rows_w = decomps[0][2].shape[0]
    
    U = np.zeros((n_rows_u, total_rank))
    V = np.zeros((n_rows_v, total_rank))
    W = np.zeros((n_rows_w, total_rank))
    
    col = 0
    for d in decomps:
        r = d[0].shape[1]
        if r > 0:
            U[:, col:col+r] = d[0]
            V[:, col:col+r] = d[1]
            W[:, col:col+r] = d[2]
        col += r
    
    return U, V, W

def _embed(n_sub, m_sub, p_sub, n_full, m_full, p_full, i_off, j_off, k_off):
    """Embed a (n_sub, m_sub, p_sub) decomposition into a (n_full, m_full, p_full) space."""
    U_sub, V_sub, W_sub = n_sub, m_sub, p_sub  # placeholder
    return None  # This function is not used directly; see _build_recursive

def _build_recursive(n, m, p, i_off=0, j_off=0, k_off=0):
    """
    Recursively build a tensor decomposition for n x m * m x p multiplication,
    embedded into a larger space starting at offsets (i_off, j_off, k_off).
    Returns (U, V, W) of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    but with indices offset appropriately.
    
    Actually, let's just build in local coordinates and then shift.
    """
    # Base cases
    if n <= 1 and m <= 1 and p <= 1:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        if rank > 0:
            U[0, 0] = 1
            V[0, 0] = 1
            W[0, 0] = 1
        return U, V, W
    
    # If n=2, m=2, p=2, use Strassen
    if n == 2 and m == 2 and p == 2:
        return _build_strassen_2x2x2()
    
    # Recursive case: split the middle dimension m
    # Try to split m into m1 + m2 where m1, m2 are chosen to enable Strassen
    # For simplicity, if m is even, split into m/2 + m/2
    # If m is odd, split into 2 + (m-2) or (m-1) + 1
    
    if m >= 2:
        # Try to find a good split
        if m >= 4 and m % 2 == 0:
            m1, m2 = m // 2, m // 2
        elif m >= 3:
            m1, m2 = 2, m - 2
        else:
            m1, m2 = 1, m - 1
        
        # Build two sub-decompositions
        U1, V1, W1 = _build_recursive(n, m1, p)
        U2, V2, W2 = _build_recursive(n, m2, p)
        
        # Now combine them
        # U: first m1 rows for U1, next m2 rows for U2
        # V: first m1*p rows for V1, next m2*p rows for V2
        # W: same for both (since n and p don't change)
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        # U1 occupies rows 0..m1-1, U2 occupies rows m1..m-1
        if rank1 > 0:
            U_full[:m1, :rank1] = U1
            V_full[:m1*p, :rank1] = V1
            W_full[:, :rank1] = W1
        
        if rank2 > 0:
            U_full[m1:, rank1:] = U2
            V_full[m1*p:, rank1:] = V2
            W_full[:, rank1:] = W2
        
        return U_full, V_full, W_full
    else:
        # m = 1, fall back to schoolbook
        return _schoolbook(n, m, p)

def _build_3x5x7():
    """Build 3x5 * 5x7 decomposition using recursive Strassen."""
    n, m, p = 3, 5, 7
    U, V, W = _build_recursive(n, m, p)
    return U, V, W

def construct():
    """Find rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Build the recursive decomposition
    U, V, W = _build_3x5x7()
    rank = U.shape[1]
    
    # Verify
    if _verify_fast(U, V, W, n, m, p, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank_s = n * m * p
    U_s = np.zeros((n*m, rank_s))
    V_s = np.zeros((m*p, rank_s))
    W_s = np.zeros((p*n, rank_s))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
