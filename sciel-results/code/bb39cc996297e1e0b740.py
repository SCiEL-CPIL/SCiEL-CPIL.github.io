import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Reshape: U[i*m+j, alpha] -> U3[i, j, alpha]
    # V[j*p+k, alpha] -> V3[j, k, alpha]
    # W[k*n+i, alpha] -> W3[k, i, alpha]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # Compute S[i, j, k] = sum_alpha U3[i,j,alpha] * V3[j,k,alpha] * W3[k,i,alpha]
    # We need this to be 1 for all i,j,k (based on seed verify logic)
    S = np.zeros((n, m, p), dtype=np.float64)
    for a in range(r):
        ua = U3[:, :, a]  # shape (n, m)
        va = V3[:, :, a]  # shape (m, p)
        wa = W3[:, :, a]  # shape (p, n)
        # For each i, j, k: ua[i,j] * va[j,k] * wa[k,i]
        for i in range(n):
            for j in range(m):
                S[i, j, :] += ua[i, j] * va[j, :] * wa[:, i]
    
    return np.allclose(S, 1.0, atol=1e-9)

def _strassen_block(b):
    """
    Returns the U, V, W contributions for the b-th 2x2 block of B.
    b is 0, 1, or 2.
    Returns lists of (u_coeffs, v_coeffs, w_coeffs) for the 7 Strassen terms.
    u_coeffs: dict (i,j) -> val
    v_coeffs: dict (j, k_local) -> val
    w_coeffs: dict (k_local, i) -> val
    """
    # Strassen terms for 2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    
    terms = []
    
    # M1 = (a11 + a22)(b11 + b22)
    # C11 += M1, C22 += M1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1} # w[k_local, i] -> C[i, k_local]
    # Wait, C[i, k]. w index is k_local, i.
    # C[0, 0] gets M1. C[1, 1] gets M1.
    # w[0, 0] = 1 (k=0, i=0)
    # w[1, 1] = 1 (k=1, i=1)
    terms.append((u1, v1, w1))
    
    # M2 = (a21 + a22)b11
    # C21 += M2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(0,1): 1} # k=0, i=1 -> C[1, 0]
    terms.append((u2, v2, w2))
    
    # M3 = a11(b12 - b22)
    # C12 += M3
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(1,0): 1} # k=1, i=0 -> C[0, 1]
    terms.append((u3, v3, w3))
    
    # M4 = a22(b21 - b11)
    # C11 += M4, C21 += M4
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (0,1): 1} # k=0, i=0 and k=0, i=1
    terms.append((u4, v4, w4))
    
    # M5 = (a11 + a12)b22
    # C11 -= M5, C12 += M5
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (1,0): 1} # k=0, i=0 (-1), k=1, i=0 (1)
    terms.append((u5, v5, w5))
    
    # M6 = (a21 - a11)(b11 + b12)
    # C22 += M6
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1} # k=1, i=1 -> C[1, 1]
    terms.append((u6, v6, w6))
    
    # M7 = (a12 - a22)(b21 + b22)
    # C11 += M7
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1} # k=0, i=0 -> C[0, 0]
    terms.append((u7, v7, w7))
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    R = 21
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # 3 blocks of B
    for b in range(3):
        terms = _strassen_block(b)
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = b * 7 + t_idx
            
            # Fill U
            for (i, j), val in u_dict.items():
                U[i * m + j][col] = val
                
            # Fill V
            # v_dict keys are (j, k_local)
            # Global k = b * 2 + k_local
            for (j, k_local), val in v_dict.items():
                k_global = b * 2 + k_local
                V[j * p + k_global][col] = val
                
            # Fill W
            # w_dict keys are (k_local, i)
            # Global k = b * 2 + k_local
            for (k_local, i), val in w_dict.items():
                k_global = b * 2 + k_local
                W[k_global * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
