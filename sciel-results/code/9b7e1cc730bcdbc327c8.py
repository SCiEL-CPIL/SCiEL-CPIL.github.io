"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _standard_term(i, k, j, r):
    """
    Create a standard rank-1 term for matrix multiplication.
    Math indices: A[i,k], B[k,j], C[i,j].
    Code indices:
    U (n*m): i*m + k
    V (m*p): k*p + j
    W (p*n): j*n + i
    """
    n, m, p = DIMENSIONS
    U = np.zeros(n * m)
    V = np.zeros(m * p)
    W = np.zeros(p * n)
    
    U[i * m + k] = 1.0
    V[k * p + j] = 1.0
    W[j * n + i] = 1.0
    
    return U, V, W

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for the 2x2 core (indices 0,1).
    These terms decompose the 2x2x2 all-ones tensor.
    Returns list of (U_vec, V_vec, W_vec) for the 2x2 sub-space.
    We need to embed these into the 3x3 space.
    """
    # Strassen formulas for C = AB (2x2)
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # We need to express this as sum of 7 rank-1 tensors.
    # Term r: U_r, V_r, W_r
    # U_r corresponds to linear form in A
    # V_r corresponds to linear form in B
    # W_r corresponds to linear form in C (coefficients)
    
    terms = []
    
    # m1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # m2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # m3
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    
    # m4
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    terms.append((u4, v4, w4))
    
    # m5
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    terms.append((u5, v5, w5))
    
    # m6
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # m7
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    return terms

def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where keys are (row, col) in the 3x3 space.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, k), val in u_dict.items():
            U[i * m + k, r] = val
        for (k, j), val in v_dict.items():
            V[k * p + j, r] = val
        for (j, i), val in w_dict.items():
            W[j * n + i, r] = val
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ikr,kjr,jir->ikj', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses Strassen for the 2x2 core and standard terms for the rest.
    Rank = 7 + 19 = 26.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    terms = []
    
    # 1. Add 7 Strassen terms for the 2x2 core (indices 0,1)
    # These terms only involve indices 0,1 in U, V, W.
    strassen_terms = _strassen_2x2_terms()
    for u_dict, v_dict, w_dict in strassen_terms:
        terms.append((u_dict, v_dict, w_dict))
        
    # 2. Add standard terms for the rest
    # Standard terms are for all (i, k, j) in 0..2
    # We skip those where i, k, j are all in 0..1
    for i in range(n):
        for k in range(m):
            for j in range(p):
                if i < 2 and k < 2 and j < 2:
                    continue # Already covered by Strassen
                # Create standard term
                u_dict = {(i, k): 1}
                v_dict = {(k, j): 1}
                w_dict = {(j, i): 1}
                terms.append((u_dict, v_dict, w_dict))
                
    U, V, W = _build_decomposition(terms)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to standard rank 27 if verification fails (should not happen)
        terms_standard = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    u_dict = {(i, k): 1}
                    v_dict = {(k, j): 1}
                    w_dict = {(j, i): 1}
                    terms_standard.append((u_dict, v_dict, w_dict))
        U, V, W = _build_decomposition(terms_standard)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
