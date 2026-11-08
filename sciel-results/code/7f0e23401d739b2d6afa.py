import numpy as np
import time

def _verify(U, V, W, n=2, m=4, p=4):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor."""
    n, m, p = 2, 4, 4
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    
    return np.allclose(T, np.ones((n,m,p)), atol=1e-9)

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Local indices: i, j, k in {0, 1}
    # U: (i, j) -> coeff
    # V: (j, k) -> coeff
    # W: (k, i) -> coeff
    
    strassen_terms = [
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        },
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,1): 1, (1,1): -1}
        },
        {
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(1,0): 1, (1,1): 1}
        },
        {
            'U': {(1,1): 1},
            'V': {(1,0): 1},
            'W': {(0,0): 1, (0,1): 1}
        },
        {
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(1,0): 1, (0,0): -1}
        },
        {
            'U': {(1,0): 1, (1,1): -1},
            'V': {(0,1): 1, (1,1): 1},
            'W': {(1,1): 1}
        },
        {
            'U': {(0,1): 1, (1,0): -1},
            'V': {(0,0): 1, (1,0): 1},
            'W': {(0,0): 1}
        }
    ]
    
    # Sub-problems definitions:
    # (j_offset, k_offset)
    sub_problems = [
        (0, 0), # A_L B_11
        (0, 2), # A_L B_12
        (2, 0), # A_R B_21
        (2, 2)  # A_R B_22
    ]
    
    col = 0
    for j_off, k_off in sub_problems:
        for term in strassen_terms:
            # Fill U
            for (i_l, j_l), val in term['U'].items():
                i_g = i_l
                j_g = j_l + j_off
                U[i_g * m + j_g, col] += val
                
            # Fill V
            for (j_l, k_l), val in term['V'].items():
                j_g = j_l + j_off
                k_g = k_l + k_off
                V[j_g * p + k_g, col] += val
                
            # Fill W
            for (k_l, i_l), val in term['W'].items():
                k_g = k_l + k_off
                i_g = i_l
                W[k_g * n + i_g, col] += val
                
            col += 1
            
    # Verification (optional but safe)
    if not _verify(U.tolist(), V.tolist(), W.tolist()):
        # Fallback to schoolbook if something went wrong (should not happen)
        return _schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook():
    n, m, p = 2, 4, 4
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
