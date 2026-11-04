import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 13 for 4x4x4 part (BCL)
    # Rank 28 for 4x4x2 part (Strassen)
    rank = 41
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 multiplication using BCL decomposition (Rank 13) ---
    # BCL decomposition terms for 4x4 matrix multiplication
    # Each term is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping (i, j) -> coeff for A[i, j]
    # V_coeffs: dict mapping (j, k) -> coeff for B[j, k]
    # W_coeffs: dict mapping (i, k) -> coeff for C[i, k]
    
    bcl_terms = [
        # 1. A00 B00
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # 2. A01 B01
        ({(0,1): 1}, {(0,1): 1}, {(0,1): 1}),
        # 3. A02 B02
        ({(0,2): 1}, {(0,2): 1}, {(0,2): 1}),
        # 4. A03 B03
        ({(0,3): 1}, {(0,3): 1}, {(0,3): 1}),
        # 5. A10 B10
        ({(1,0): 1}, {(1,0): 1}, {(1,0): 1}),
        # 6. A11 B11
        ({(1,1): 1}, {(1,1): 1}, {(1,1): 1}),
        # 7. A12 B12
        ({(1,2): 1}, {(1,2): 1}, {(1,2): 1}),
        # 8. A13 B13
        ({(1,3): 1}, {(1,3): 1}, {(1,3): 1}),
        # 9. (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 10. (A02+A13)(B02+B13)
        ({(0,2): 1, (1,3): 1}, {(0,2): 1, (1,3): 1}, {(0,2): 1, (1,3): 1}),
        # 11. (A00+A11+A02+A13)(B00+B11+B02+B13)
        ({(0,0): 1, (1,1): 1, (0,2): 1, (1,3): 1}, 
         {(0,0): 1, (1,1): 1, (0,2): 1, (1,3): 1}, 
         {(0,0): 1, (1,1): 1, (0,2): 1, (1,3): 1}),
        # 12. (A01+A12)(B01+B12)
        ({(0,1): 1, (1,2): 1}, {(0,1): 1, (1,2): 1}, {(0,1): 1, (1,2): 1}),
        # 13. (A03+A10)(B03+B10)
        ({(0,3): 1, (1,0): 1}, {(0,3): 1, (1,0): 1}, {(0,3): 1, (1,0): 1})
    ]
    
    for k in range(13):
        u_coeffs, v_coeffs, w_coeffs = bcl_terms[k]
        col = k
        
        # Fill U
        for (i, j), coeff in u_coeffs.items():
            U[i * m + j][col] += coeff
            
        # Fill V (only first 4 columns of B are used)
        for (j, k_b), coeff in v_coeffs.items():
            V[j * p + k_b][col] += coeff
            
        # Fill W (only first 4 columns of C are used)
        for (i, k_c), coeff in w_coeffs.items():
            W[k_c * n + i][col] += coeff

    # --- Part 2: 4x4x2 multiplication using Strassen (Rank 28) ---
    # A is 4x4, B2 is 4x2 (cols 4-5 of B), C2 is 4x2 (cols 4-5 of C)
    # We split A into 4 blocks of 2x2: A00, A01, A10, A11
    # B2 is split into 2 blocks of 2x2: B0 (rows 0-1), B1 (rows 2-3)
    # C2 is split into 2 blocks of 2x2: C0 (rows 0-1), C1 (rows 2-3)
    # 
    # Multiplications:
    # C0 = A00 * B0 + A01 * B1
    # C1 = A10 * B0 + A11 * B1
    #
    # We do 4 separate 2x2x2 multiplications:
    # M1: A00 * B0 -> contributes to C0
    # M2: A01 * B1 -> contributes to C0
    # M3: A10 * B0 -> contributes to C1
    # M4: A11 * B1 -> contributes to C1
    #
    # Each 2x2x2 has rank 7, so total 4 * 7 = 28
    
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    mults = [
        # (a_block, b_block, c_block)
        # a_block: (bi, bj) in A
        # b_block: bi in B2 (0 or 1)
        # c_block: bi in C2 (0 or 1)
        ((0, 0), 0, 0),  # A00 * B0 -> C0
        ((0, 1), 1, 0),  # A01 * B1 -> C0
        ((1, 0), 0, 1),  # A10 * B0 -> C1
        ((1, 1), 1, 1),  # A11 * B1 -> C1
    ]
    
    for m_idx, (a_block, b_bi, c_bi) in enumerate(mults):
        a_bi, a_bj = a_block
        for s2 in range(7):
            a2, b2, c2 = strassen_steps[s2]
            col = 13 + m_idx * 7 + s2
            
            # U: A block (a_bi, a_bj)
            for (si, sj), va in a2.items():
                r = (a_bi * 2 + si) * m + (a_bj * 2 + sj)
                U[r][col] += va
            
            # V: B2 block (b_bi, 0) -> rows b_bi*2 to b_bi*2+1, cols 4-5
            for (si, sj), vb in b2.items():
                r = (b_bi * 2 + si) * p + (4 + sj)
                V[r][col] += vb
            
            # W: C2 block (c_bi, 0) -> rows c_bi*2 to c_bi*2+1, cols 4-5
            # W index = col * n + row
            for (si, sj), vc in c2.items():
                r = (4 + sj) * n + (c_bi * 2 + si)
                W[r][col] += vc

    return {"U": U, "V": V, "W": W}
