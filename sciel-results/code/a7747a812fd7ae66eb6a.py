import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for block q: 
    #   e is at V[0, 2q], f is at V[0, 2q+1]
    #   g is at V[1, 2q], h is at V[1, 2q+1]
    # W indices for block q:
    #   C[0,0] is at W[2q, 0] -> index 2q*2 + 0 = 4q
    #   C[0,1] is at W[2q+1, 0] -> index (2q+1)*2 + 0 = 4q+2
    #   C[1,0] is at W[2q, 1] -> index 2q*2 + 1 = 4q+1
    #   C[1,1] is at W[2q+1, 1] -> index (2q+1)*2 + 1 = 4q+3
    
    # Strassen terms
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # Coefficients for U (a, b, c, d)
    U_coeffs = [
        [1, 0, 1, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Coefficients for V (e, f, g, h)
    V_coeffs = [
        [1, 1, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Contributions to W (C00, C01, C10, C11)
    # Order: C00, C01, C10, C11
    W_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 0, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    for q in range(4):
        for s in range(7):
            col = 7 * q + s
            
            # Fill U
            for u_idx in range(4):
                U[u_idx][col] = U_coeffs[s][u_idx]
                
            # Fill V
            # e -> V[0, 2q], f -> V[0, 2q+1], g -> V[1, 2q], h -> V[1, 2q+1]
            U_V_map = [0, 1, 2, 3] # 0:e, 1:f, 2:g, 3:h
            V_row_map = [
                0 * 8 + 2 * q,     # e
                0 * 8 + (2 * q + 1), # f
                1 * 8 + 2 * q,     # g
                1 * 8 + (2 * q + 1)  # h
            ]
            for v_idx in range(4):
                V[V_row_map[v_idx]][col] = V_coeffs[s][v_idx]
                
            # Fill W
            # C00 -> W[4q], C01 -> W[4q+2], C10 -> W[4q+1], C11 -> W[4q+3]
            W_row_map = [
                4 * q,       # C00
                4 * q + 2,   # C01
                4 * q + 1,   # C10
                4 * q + 3    # C11
            ]
            for w_idx in range(4):
                W[W_row_map[w_idx]][col] = W_coeffs[s][w_idx]
                
    return {"U": U, "V": V, "W": W}
