import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    R = 88
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    col = 0
    
    # Part 1: 1x4x8 (Row 2 of A)
    for k in range(p):
        for j in range(m):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Part 2: 2x4x8 (Rows 0,1 of A)
    # Strassen terms definition
    # Each term is a list of (index, value) for U, V, W
    # Indices are local (0,1)
    
    strassen_terms = [
        # Term 1
        ([(0,0,1), (1,1,1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # Term 2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # Term 3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # Term 4
        ([(1,1,1), (1,0,-1)], [(1,0,1)], [(0,0,1), (1,0,1)]),
        # Term 5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # Term 6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # Term 7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for k_offset in [0, 4]:
        for j_offset in [0, 2]:
            for k_local_offset in [0, 2]:
                for term in strassen_terms:
                    u_data, v_data, w_data = term
                    
                    for i_l, j_l, val in u_data:
                        i_g = i_l # i_offset is 0
                        j_g = j_offset + j_l
                        U[i_g * m + j_g, col] = val
                        
                    for j_l, k_l, val in v_data:
                        j_g = j_offset + j_l
                        k_g = k_offset + k_local_offset + k_l
                        V[j_g * p + k_g, col] = val
                        
                    for k_l, i_l, val in w_data:
                        k_g = k_offset + k_local_offset + k_l
                        i_g = i_l
                        W[k_g * n + i_g, col] = val
                        
                    col += 1

    return {"U": U, "V": V, "W": W}
