import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen on 2x2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (local 2x2)
    # Indices: 00, 01, 10, 11
    strassen_A = [
        [1, 0, 0, 1], # M0
        [0, 0, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 0, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0], # M5
        [0, 1, 0, -1]  # M6
    ]
    
    # Strassen coefficients for B (local 2x2)
    strassen_B = [
        [1, 0, 0, 1], # M0
        [1, 0, 0, 0], # M1
        [0, 1, 0, -1], # M2
        [-1, 0, 1, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, 1]  # M6
    ]
    
    # W coefficients: w[i][k_loc] for each t
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    w_coeffs = [
        [1, 0, 1, -1, 1], # t=0: C00=1, C01=0, C10=0, C11=1 -> Wait, format is [i=0,k=0], [i=0,k=1], [i=1,k=0], [i=1,k=1]
        # Let's re-verify w_coeffs structure.
        # w_coeffs[t][i*2 + k_loc]
    ]
    
    # t=0: C00(1), C11(1)
    # t=1: C10(1), C11(-1)
    # t=2: C01(1), C11(1)
    # t=3: C00(1), C10(1)
    # t=4: C00(-1), C01(1)
    # t=5: C11(1)
    # t=6: C00(1)
    
    w_coeffs = [
        [1, 0, 0, 1], # t=0
        [0, 0, 1, -1], # t=1
        [0, 1, 0, 1], # t=2
        [1, 0, 1, 0], # t=3
        [-1, 1, 0, 0], # t=4
        [0, 0, 0, 1], # t=5
        [1, 0, 0, 0]  # t=6
    ]
    
    # 6 blocks: b = 0..5
    # b = 3*s + k_block
    # s in 0..1, k_block in 0..2
    for b in range(6):
        s = b // 3
        k_block = b % 3
        
        for t in range(7):
            c = 7 * b + t
            
            # Fill U
            u_coeff = strassen_A[t]
            for i in range(2):
                for j_loc in range(2):
                    idx_A = i * 2 + j_loc
                    coeff = u_coeff[idx_A]
                    if coeff != 0:
                        global_i = i
                        global_j = 2 * s + j_loc
                        U[global_i * m + global_j][c] = coeff
                        
            # Fill V
            v_coeff = strassen_B[t]
            for j_loc in range(2):
                for k_loc in range(2):
                    idx_B = j_loc * 2 + k_loc
                    coeff = v_coeff[idx_B]
                    if coeff != 0:
                        global_j = 2 * s + j_loc
                        global_k = 2 * k_block + k_loc
                        V[global_j * p + global_k][c] = coeff
                        
            # Fill W
            w_coeff = w_coeffs[t]
            for i in range(2):
                for k_loc in range(2):
                    idx_W = i * 2 + k_loc
                    coeff = w_coeff[idx_W]
                    if coeff != 0:
                        global_k = 2 * k_block + k_loc
                        global_i = i
                        W[global_k * n + global_i][c] = coeff
                        
    return {"U": U, "V": V, "W": W}
