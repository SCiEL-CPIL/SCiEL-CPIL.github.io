import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 112 decomposition
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # N (A block) and P (B block) are 2x2.
    # M (C block) is 2x2.
    # Indices 0..3 for 2x2 matrix: 00, 01, 10, 11
    
    # M1 = (N00 + N11)(P00 + P11)
    # M2 = (N10 + N11) P00
    # M3 = N00 (P10 - P11)
    # M4 = N11 (P01 - P00)
    # M5 = (N00 + N10) P11
    # M6 = (N01 - N00)(P00 + P10)
    # M7 = (N10 - N11)(P01 + P11)
    
    # U coeffs (for N), V coeffs (for P), W coeffs (for M)
    # W coeffs are for M00, M01, M10, M11
    strassen = [
        # U (N00, N01, N10, N11), V (P00, P01, P10, P11), W (M00, M01, M10, M11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, -1, 0]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, -1, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, -1]),  # M4
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),   # M5
        ([1, -1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]),   # M6
        ([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 1, 1]),   # M7
    ]
    
    col = 0
    for b in range(2): # Block of B (Left/Right)
        for bp in range(4): # Block product in 4x4 (00, 01, 10, 11)
            i_block = bp // 2
            k_block = bp % 2
            for j_block in range(2): # Summation over j blocks
                for s in range(7): # Strassen step
                    u_coeffs = strassen[s][0]
                    v_coeffs = strassen[s][1]
                    w_coeffs = strassen[s][2]
                    
                    # A block is A[i_block, j_block]
                    # B block is B[j_block, k_block] in block b
                    # C block is C[i_block, k_block] in block b
                    
                    # Fill U
                    for r in range(2):
                        for c in range(2):
                            coeff = u_coeffs[r*2+c]
                            if coeff != 0:
                                i = 2 * i_block + r
                                j = 2 * j_block + c
                                U[i * m + j][col] = coeff
                                
                    # Fill V
                    for r in range(2):
                        for c in range(2):
                            coeff = v_coeffs[r*2+c]
                            if coeff != 0:
                                j = 2 * j_block + r
                                k = 4 * b + 2 * k_block + c
                                V[j * p + k][col] = coeff
                                
                    # Fill W
                    for r in range(2):
                        for c in range(2):
                            coeff = w_coeffs[r*2+c]
                            if coeff != 0:
                                i = 2 * i_block + r
                                k = 4 * b + 2 * k_block + c
                                W[k * n + i][col] = coeff
                                
                    col += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
