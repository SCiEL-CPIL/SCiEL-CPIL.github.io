"""Matrix multiplication 4,4,9 schoolbook decomposition.

Approach: Direct term-by-term summation (schoolbook).
Total rank: 4 * 4 * 9 = 144.
This is a new approach family (non-recursive, direct) compared to the previous Strassen-based design.
"""

import numpy as np

DIMENSIONS = [4, 4, 9]


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses schoolbook decomposition with rank 144.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(list(u_vec))
        V_cols.append(list(v_vec))
        W_cols.append(list(w_vec))
    
    # Schoolbook: C[i, k] = sum_j A[i, j] * B[j, k]
    # Each term is A[i, j] * B[j, k] contributing to C[i, k]
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                u_vec[u_idx] = 1
                v_vec[v_idx] = 1
                w_vec[w_idx] = 1
                
                add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = np.array(U_cols).T.tolist()
    V = np.array(V_cols).T.tolist()
    W = np.array(W_cols).T.tolist()
    
    return {"U": U, "V": V, "W": W}
