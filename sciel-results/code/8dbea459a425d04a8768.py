import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A21)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to U, V, W such that sum U*V*W = T
    # T[i,j,k] = 1 for all i,j,k in 0..1
    
    # This is a known rank-7 decomposition.
    # U, V, W are 4x7 matrices.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: M1
    # A_part: A11+A22 -> U indices 0,3
    # B_part: B11+B22 -> V indices 0,3
    # C_part: C11, C22 -> W indices 0,3
    # C11 = M1 + ...
    # C22 = M1 - ...
    
    # We define the factors for each term r
    # r=0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2
    # A_part: A21+A22 -> U indices 2,3
    # B_part: B11 -> V index 0
    # C_part: C21, C22
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C21 gets +, C22 gets -? 
    # Strassen: C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So W for M2 should be +1 for C21 (idx 2) and -1 for C22 (idx 3).
    
    # r=2: M3
    # A_part: A11 -> U index 0
    # B_part: B12-B22 -> V indices 1, 3 (with -1)
    # C_part: C12, C22
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C12 gets +, C22 gets +
    
    # r=3: M4
    # A_part: A22 -> U index 3
    # B_part: B21-B11 -> V indices 2, 0 (with -1)
    # C_part: C11, C21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11 gets +, C21 gets +
    
    # r=4: M5
    # A_part: A11+A21 -> U indices 0, 2
    # B_part: B22 -> V index 3
    # C_part: C12, C11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1 # C12 gets +, C11 gets -
    
    # r=5: M6
    # A_part: A21-A11 -> U indices 2, 0 (with -1)
    # B_part: B11+B12 -> V indices 0, 1
    # C_part: C22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22 gets +
    
    # r=6: M7
    # A_part: A12-A22 -> U indices 1, 3 (with -1)
    # B_part: B21+B22 -> V indices 2, 3
    # C_part: C11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11 gets +
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of (U, V, W) tuples into one.
    Each tuple must correspond to a sub-problem that sums to the full tensor.
    This function assumes the decomps are already scaled/aligned to the full n,m,p indices.
    """
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for U_d, V_d, W_d in decomps:
        U[:, col:col+U_d.shape[1]] = U_d
        V[:, col:col+V_d.shape[1]] = V_d
        W[:, col:col+W_d.shape[1]] = W_d
        col += U_d.shape[1]
        
    return U, V, W

def _embed(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, n_full, m_full, p_full, i_off, j_off, k_off):
    """
    Embed a sub-decomposition into the full n,m,p space.
    i_off, j_off, k_off are the offsets in the full indices.
    """
    n, m, p = n_full, m_full, p_full
    rank = U_sub.shape[1]
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for i in range(n_sub):
            for j in range(m_sub):
                val = U_sub[i * m_sub + j, r]
                if val != 0:
                    U[(i + i_off) * m + (j + j_off), r] = val
        for j in range(m_sub):
            for k in range(p_sub):
                val = V_sub[j * p_sub + k, r]
                if val != 0:
                    V[(j + j_off) * p + (k + k_off), r] = val
        for k in range(p_sub):
            for i in range(n_sub):
                val = W_sub[k * n_sub + i, r]
                if val != 0:
                    W[(k + k_off) * n + (i + i_off), r] = val
    return U, V, W

def _get_decomp(n, m, p):
    """
    Recursively get decomposition for n x m x p.
    """
    # Base cases
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2_factors()
    
    if n == 1 or m == 1 or p == 1:
        return _schoolbook(n, m, p)
        
    # Recursive case
    # Strategy: Split m into 2 and m-2 if m > 2
    if m > 2:
        m1, m2 = 2, m - 2
        # Sub-problem 1: n x m1 x p
        U1, V1, W1 = _get_decomp(n, m1, p)
        # Embed at j_off=0
        U1_e, V1_e, W1_e = _embed(U1, V1, W1, n, m1, p, n, m, p, 0, 0, 0)
        
        # Sub-problem 2: n x m2 x p
        U2, V2, W2 = _get_decomp(n, m2, p)
        # Embed at j_off=m1
        U2_e, V2_e, W2_e = _embed(U2, V2, W2, n, m2, p, n, m, p, 0, m1, 0)
        
        return _combine_decompositions([(U1_e, V1_e, W1_e), (U2_e, V2_e, W2_e)], n, m, p)
        
    # If m == 2, try splitting n
    if m == 2 and n > 2:
        n1, n2 = 2, n - 2
        U1, V1, W1 = _get_decomp(n1, m, p)
        U1_e, V1_e, W1_e = _embed(U1, V1, W1, n1, m, p, n, m, p, 0, 0, 0)
        
        U2, V2, W2 = _get_decomp(n2, m, p)
        U2_e, V2_e, W2_e = _embed(U2, V2, W2, n2, m, p, n, m, p, n1, 0, 0)
        
        return _combine_decompositions([(U1_e, V1_e, W1_e), (U2_e, V2_e, W2_e)], n, m, p)
        
    # If m == 2 and n == 2, try splitting p
    if m == 2 and n == 2 and p > 2:
        p1, p2 = 2, p - 2
        U1, V1, W1 = _get_decomp(n, m, p1)
        U1_e, V1_e, W1_e = _embed(U1, V1, W1, n, m, p1, n, m, p, 0, 0, 0)
        
        U2, V2, W2 = _get_decomp(n, m, p2)
        U2_e, V2_e, W2_e = _embed(U2, V2, W2, n, m, p2, n, m, p, 0, 0, p1)
        
        return _combine_decompositions([(U1_e, V1_e, W1_e), (U2_e, V2_e, W2_e)], n, m, p)
        
    # Fallback: Schoolbook
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U, V, W = _get_decomp(n, m, p)
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
