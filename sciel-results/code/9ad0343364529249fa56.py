import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen on 2x2x6 blocks
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(e-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Coefficients for A (a,b,c,d) -> indices 0,1,2,3 in local 2x2
    # Coefficients for B (e,f,g,h) -> indices 0,1,2,3 in local 2x2
    # Note: B local indices map to e,f,g,h as:
    # e=0, f=1, g=2, h=3 (row-major 2x2)
    
    A_coeffs = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    B_coeffs = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 0, 1, -1],# M3: g-h
        [1, -1, 0, 0],# M4: e-f
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # Output coefficients for C11, C12, C21, C22
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4 + M6
    C_coeffs = [
        [1, 0, 0, -1, 1, 0, -1], # M1..M7 for C11
        [0, 0, 1, 0, 0, 0, 1],   # for C12
        [0, 1, 0, 0, 0, 1, 0],   # for C21
        [1, 0, -1, 1, 0, 1, 0]   # for C22
    ]
    
    col = 0
    # Two parts of A (j=0,1 and j=2,3)
    for part in range(2):
        j_offset = part * 2
        
        # Three blocks of B (k=0,1; 2,3; 4,5)
        for t in range(3):
            k_offset = t * 2
            
            for r in range(7):
                # U: n*m = 2*4 = 8
                # Local A indices: i in 0,1; j_local in 0,1
                for i in range(n):
                    for j_local in range(2):
                        j_global = j_offset + j_local
                        u_idx = i * m + j_global
                        U[u_idx, col] = A_coeffs[r][i * 2 + j_local]
                
                # V: m*p = 4*6 = 24
                # Local B indices: j_local in 0,1; k_local in 0,1
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = j_offset + j_local
                        k_global = k_offset + k_local
                        v_idx = j_global * p + k_global
                        V[v_idx, col] = B_coeffs[r][j_local * 2 + k_local]
                
                # W: p*n = 6*2 = 12
                # Output C indices: i in 0,1; k_local in 0,1
                # W index = k_global * n + i
                for i in range(n):
                    for k_local in range(2):
                        k_global = k_offset + k_local
                        w_idx = k_global * n + i
                        # C_coeffs[r] gives coefficients for C11, C12, C21, C22
                        # C11 corresponds to i=0, k_local=0
                        # C12 corresponds to i=0, k_local=1
                        # C21 corresponds to i=1, k_local=0
                        # C22 corresponds to i=1, k_local=1
                        c_idx = i * 2 + k_local
                        W[w_idx, col] = C_coeffs[c_idx][r]
                        
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
