import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 components
    # A entries: 0, 1, 2, 3 (row-major: a,b,c,d)
    # B entries: 0, 1, 2, 3 (row-major: e,f,g,h)
    # C entries: 0, 1, 2, 3 (row-major: c11, c12, c21, c22)
    
    # U_s: vector of coeffs for A
    # V_s: vector of coeffs for B
    # W_s: vector of coeffs for C
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1], # M6
        [0, 1, -1, 0]  # M7
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1
        [0, 0, 0, 1], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 1, 0],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1 -> C11, C22
        [0, 0, 1, -1], # M2 -> C21, C22
        [0, 1, 0, 1],  # M3 -> C12, C22
        [1, 0, 1, 0],  # M4 -> C11, C21
        [-1, 1, 0, 0], # M5 -> C11, C12
        [0, 0, 0, 1],  # M6 -> C22
        [1, 0, 0, 0]   # M7 -> C11
    ]
    
    col = 0
    for s in range(2): # Split A and B rows
        # A block s: rows 0,1 of A_s? No, A is 2x4.
        # A_s is 2x2. 
        # A is indexed i (0,1) in n, j (0..3) in m.
        # A_s corresponds to j in {2s, 2s+1}.
        # So A_s entries are at indices:
        # a00 (i=0, j=2s), a01 (i=0, j=2s+1)
        # a10 (i=1, j=2s), a11 (i=1, j=2s+1)
        
        for t in range(3): # Split B cols
            # B block (s,t):
            # B is 4x6.
            # B_s corresponds to rows 2s, 2s+1.
            # B_t corresponds to cols 2t, 2t+1.
            # So B_{s,t} entries are at indices (j, k) in V.
            # V index is j*p + k.
            # e (j=2s, k=2t), f (j=2s, k=2t+1)
            # g (j=2s+1, k=2t), h (j=2s+1, k=2t+1)
            
            for r in range(7): # Strassen rank 7
                # Assign U
                for k_u in range(4):
                    if strassen_U[r][k_u] != 0:
                        # Map local A index k_u to global A index
                        # k_u=0 -> (0, 2s)
                        # k_u=1 -> (0, 2s+1)
                        # k_u=2 -> (1, 2s)
                        # k_u=3 -> (1, 2s+1)
                        # Global A index = i * m + j
                        if k_u == 0:
                            i, j = 0, 2 * s
                        elif k_u == 1:
                            i, j = 0, 2 * s + 1
                        elif k_u == 2:
                            i, j = 1, 2 * s
                        elif k_u == 3:
                            i, j = 1, 2 * s + 1
                        else:
                            continue
                        U[i * m + j][col] += strassen_U[r][k_u]
                        
                # Assign V
                for k_v in range(4):
                    if strassen_V[r][k_v] != 0:
                        # Map local B index k_v to global B index
                        # k_v=0 -> (2s, 2t)
                        # k_v=1 -> (2s, 2t+1)
                        # k_v=2 -> (2s+1, 2t)
                        # k_v=3 -> (2s+1, 2t+1)
                        if k_v == 0:
                            j, k = 2 * s, 2 * t
                        elif k_v == 1:
                            j, k = 2 * s, 2 * t + 1
                        elif k_v == 2:
                            j, k = 2 * s + 1, 2 * t
                        elif k_v == 3:
                            j, k = 2 * s + 1, 2 * t + 1
                        else:
                            continue
                        V[j * p + k][col] += strassen_V[r][k_v]
                        
                # Assign W
                for k_w in range(4):
                    if strassen_W[r][k_w] != 0:
                        # Map local C index k_w to global C index
                        # k_w=0 -> C(0, 2t)
                        # k_w=1 -> C(0, 2t+1)
                        # k_w=2 -> C(1, 2t)
                        # k_w=3 -> C(1, 2t+1)
                        # Global C index = k * n + i
                        if k_w == 0:
                            i, k = 0, 2 * t
                        elif k_w == 1:
                            i, k = 0, 2 * t + 1
                        elif k_w == 2:
                            i, k = 1, 2 * t
                        elif k_w == 3:
                            i, k = 1, 2 * t + 1
                        else:
                            continue
                        W[k * n + i][col] += strassen_W[r][k_w]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
