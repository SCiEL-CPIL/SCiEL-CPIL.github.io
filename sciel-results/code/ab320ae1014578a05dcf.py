"""Rewritten program for Matrix Multiplication 2,2,4 with rank 14."""

def construct():
    """
    Returns a rank-14 exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x4 matrix.
    
    U: (4, 14)
    V: (8, 14)
    W: (8, 14)
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 rank-1 terms for 2x2 * 2x2
    # Each term is a tuple (u, v, w) where u, v, w are dicts mapping index to coeff
    # Indices for 2x2:
    # U: (i,j) -> i*2+j
    # V: (j,k) -> j*2+k (local k)
    # W: (k,i) -> k*2+i (local k)
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1}),
        # M3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # M4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1}),
        # M6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Block 1: k in {0, 1}
    # V global index: j*4 + k
    # W global index: k*2 + i
    for col_offset in range(7):
        u_dict, v_dict, w_dict = strassen_terms[col_offset]
        col = col_offset
        
        for (i, j), val in u_dict.items():
            U[i * m + j][col] += val
            
        for (j, k_local), val in v_dict.items():
            k_global = k_local # 0 or 1
            V[j * p + k_global][col] += val
            
        for (k_local, i), val in w_dict.items():
            k_global = k_local # 0 or 1
            W[k_global * n + i][col] += val
            
    # Block 2: k in {2, 3}
    # V global index: j*4 + k
    # W global index: k*2 + i
    for col_offset in range(7):
        u_dict, v_dict, w_dict = strassen_terms[col_offset]
        col = 7 + col_offset
        
        for (i, j), val in u_dict.items():
            U[i * m + j][col] += val
            
        for (j, k_local), val in v_dict.items():
            k_global = 2 + k_local # 2 or 3
            V[j * p + k_global][col] += val
            
        for (k_local, i), val in w_dict.items():
            k_global = 2 + k_local # 2 or 3
            W[k_global * n + i][col] += val
            
    return {"U": U, "V": V, "W": W}
