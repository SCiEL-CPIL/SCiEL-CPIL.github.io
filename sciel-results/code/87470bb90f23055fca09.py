import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen) + 16 for 4x4x1 = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    
    # Base Strassen 2x2x2 decomposition
    # Indices for 2x2:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> k*2+i
    
    # The 7 terms of Strassen for 2x2 matrices
    # Term 0: M1 = (A00+A11)(B00+B11)
    # Term 1: M2 = (A10+A11)B00
    # Term 2: M3 = A00(B10-B11)
    # Term 3: M4 = A11(B01-B00)
    # Term 4: M5 = (A00+A01)B11
    # Term 5: M6 = (A10-A00)(B00+B01)
    # Term 6: M7 = (A01-A11)(B10+B11)
    
    base_U = [
        [1, 0, 1, 0, 0, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    
    base_V = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    base_W = [
        [1, 0, 0, 1, -1, 0, 1], # C00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (k=0, i=1)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1)
    ]
    
    # Strassen coefficients for 4x4 blocks
    # s: 0..6
    # c[s][alpha] where alpha is 0:00, 1:01, 2:10, 3:11
    # d[s][beta] where beta is 0:00, 1:01, 2:10, 3:11
    # w_out[s][K][I] where K,I are 0 or 1 (block row/col of C)
    
    coeffs_c = [
        [1, 0, 0, 1], # s=0 (M1): A00, A11
        [0, 0, 1, 1], # s=1 (M2): A10, A11
        [1, 0, 0, 0], # s=2 (M3): A00
        [0, 0, 0, 1], # s=3 (M4): A11
        [1, 1, 0, 0], # s=4 (M5): A00, A01
        [-1, 0, 1, 0],# s=5 (M6): A00, A10
        [0, 1, 0, -1] # s=6 (M7): A01, A11
    ]
    
    coeffs_d = [
        [1, 0, 0, 1], # s=0 (M1): B00, B11
        [1, 0, 0, 0], # s=1 (M2): B00
        [0, 0, 1, -1],# s=2 (M3): B10, B11
        [-1, 1, 0, 0],# s=3 (M4): B00, B01
        [0, 0, 0, 1], # s=4 (M5): B11
        [1, 1, 0, 0], # s=5 (M6): B00, B01
        [0, 0, 1, 1]  # s=6 (M7): B10, B11
    ]
    
    # Output mapping: C[K][I] += w * M_s
    # C00: M1(1) + M4(1) - M5(1) + M7(1)
    # C01: M3(1) + M5(1)
    # C10: M2(1) + M4(1)
    # C11: M1(1) - M2(1) + M3(1) + M6(1)
    
    out_w = [
        [[1, 0], [0, 0]], # s=0 (M1): C00(1), C11(1) -> Wait, indices are [K][I]
        [[0, 0], [1, -1]], # s=1 (M2): C10(1), C11(-1)
        [[0, 1], [0, 1]],  # s=2 (M3): C01(1), C11(1)
        [[1, 0], [1, 0]],  # s=3 (M4): C00(1), C10(1)
        [[-1, 1], [0, 0]], # s=4 (M5): C00(-1), C01(1)
        [[0, 0], [0, 1]],  # s=5 (M6): C11(1)
        [[1, 0], [0, 0]]   # s=6 (M7): C00(1)
    ]
    # Correction: out_w[s][K][I]
    # s=0: C00(1), C11(1) -> [[1,0], [0,1]]
    # s=1: C10(1), C11(-1) -> [[0,0], [1,-1]]
    # s=2: C01(1), C11(1) -> [[0,1], [0,1]]
    # s=3: C00(1), C10(1) -> [[1,0], [1,0]]
    # s=4: C00(-1), C01(1) -> [[-1,1], [0,0]]
    # s=5: C11(1) -> [[0,0], [0,1]]
    # s=6: C00(1) -> [[1,0], [0,0]]
    
    out_w = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    for s in range(7):
        for b in range(7):
            r = s * 7 + b
            # Fill U
            for alpha in range(4):
                if coeffs_c[s][alpha] != 0:
                    I = alpha // 2
                    J = alpha % 2
                    for i_loc in range(2):
                        for j_loc in range(2):
                            val = coeffs_c[s][alpha] * base_U[i_loc * 2 + j_loc][b]
                            if val != 0:
                                i_glob = I * 2 + i_loc
                                j_glob = J * 2 + j_loc
                                U[i_glob * m + j_glob][r] += val
                                
            # Fill V
            for beta in range(4):
                if coeffs_d[s][beta] != 0:
                    J = beta // 2
                    K = beta % 2
                    for j_loc in range(2):
                        for k_loc in range(2):
                            val = coeffs_d[s][beta] * base_V[j_loc * 2 + k_loc][b]
                            if val != 0:
                                j_glob = J * 2 + j_loc
                                k_glob = K * 2 + k_loc
                                V[j_glob * p + k_glob][r] += val
                                
            # Fill W
            for K in range(2):
                for I in range(2):
                    if out_w[s][K][I] != 0:
                        for k_loc in range(2):
                            for i_loc in range(2):
                                val = out_w[s][K][I] * base_W[k_loc * 2 + i_loc][b]
                                if val != 0:
                                    k_glob = K * 2 + k_loc
                                    i_glob = I * 2 + i_loc
                                    W[k_glob * n + i_glob][r] += val

    # --- Part 2: 4x4x1 (Rank 16) ---
    # A (4x4) * B_col (4x1) -> C_col (4x1)
    # B_col is the 5th column of B (index k=4)
    # C_col is the 5th column of C (index k=4)
    
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1 # B index: j*5 + 4
            W[4 * n + i][r] = 1 # C index: 4*4 + i
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
