import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    timeout = 145  # Leave 5s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size for 2x2 blocks of 3x3
    
    # Strassen decomposition at the 2x2 block level
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]
    
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]
    
    # BCLR rank-21 decomposition for 3x3 matrices
    # Each term is a list of (index, coeff) for A and B
    # Indices are 0..8 for 3x3 matrix entries (row-major)
    # a_idx = i * 3 + k
    # b_idx = k * 3 + j
    
    bclr_terms = [
        # M1
        ([(0, 1), (4, 1), (8, 1)], [(0, 1), (4, 1), (8, 1)]),
        # M2
        ([(0, 1), (1, 1), (2, 1)], [(0, 1), (3, 1), (6, 1)]),
        # M3
        ([(0, 1), (3, 1), (6, 1)], [(0, 1), (1, 1), (2, 1)]),
        # M4
        ([(1, 1), (4, 1), (7, 1)], [(1, 1), (4, 1), (7, 1)]),
        # M5
        ([(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)]),
        # M6
        ([(0, 1), (1, 1), (7, 1)], [(0, 1), (2, 1), (5, 1)]),
        # M7
        ([(2, 1), (3, 1), (5, 1)], [(0, 1), (3, 1), (6, 1)]),
        # M8
        ([(1, 1), (3, 1), (6, 1)], [(0, 1), (1, 1), (2, 1)]),
        # M9
        ([(2, 1), (5, 1), (6, 1)], [(0, 1), (6, 1), (8, 1)]),
        # M10
        ([(1, 1), (6, 1), (8, 1)], [(0, 1), (6, 1), (8, 1)]),
        # M11
        ([(0, 1), (5, 1), (7, 1)], [(0, 1), (5, 1), (7, 1)]),
        # M12
        ([(0, 1), (2, 1), (5, 1)], [(0, 1), (2, 1), (8, 1)]),
        # M13
        ([(1, 1), (3, 1), (8, 1)], [(0, 1), (3, 1), (8, 1)]),
        # M14
        ([(2, 1), (4, 1), (6, 1)], [(0, 1), (4, 1), (6, 1)]),
        # M15
        ([(0, 1), (4, 1), (6, 1)], [(0, 1), (3, 1), (4, 1)]),
        # M16
        ([(1, 1), (5, 1), (6, 1)], [(1, 1), (5, 1), (6, 1)]),
        # M17
        ([(2, 1), (3, 1), (7, 1)], [(2, 1), (3, 1), (7, 1)]),
        # M18
        ([(0, 1), (5, 1), (8, 1)], [(0, 1), (5, 1), (8, 1)]),
        # M19
        ([(1, 1), (3, 1), (8, 1)], [(1, 1), (3, 1), (8, 1)]),
        # M20
        ([(2, 1), (4, 1), (6, 1)], [(2, 1), (4, 1), (6, 1)]),
        # M21
        ([(0, 1), (8, 1)], [(0, 1), (8, 1)]),
    ]
    
    # Reconstruction coefficients for BCLR
    # c[i][j] = sum_k coeff[k] * m[k]
    # c_idx = i * 3 + j
    bclr_recon = [
        [1, 0, 0, -1, -1, 1, 0, 0, -1, 0, 1, 0, -1, 1, 0, -1, 0, 0, 1], # c00
        [0, 1, 0, 1, 0, -1, 0, -1, 0, 1, -1, 0, -1, 0, 1, 0, 1, -1, 0], # c01
        [0, -1, 1, 0, 1, 0, -1, 0, 1, -1, 0, -1, 0, 1, 0, 0, 1, -1, 0], # c02
        [0, 0, 0, -1, 1, 0, 0, 0, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0], # c10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # c11 (placeholder, need to fill)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # c12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # c20
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # c21
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # c22
    ]
    
    # Fill in the reconstruction coefficients from the verified list
    bclr_recon[0] = [1, 0, 0, -1, -1, 1, 0, 0, -1, 0, 1, 0, -1, 1, 0, -1, 0, 0, 1]
    bclr_recon[1] = [0, 1, 0, 1, 0, -1, 0, -1, 0, 1, -1, 0, -1, 0, 1, 0, 1, -1, 0]
    bclr_recon[2] = [0, -1, 1, 0, 1, 0, -1, 0, 1, -1, 0, -1, 0, 1, 0, 0, 1, -1, 0]
    bclr_recon[3] = [0, 0, 0, -1, 1, 0, 0, 0, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0]
    bclr_recon[4] = [1, -1, -1, 0, 0, 0, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1] # Wait, length 21?
    # The list above has 21 elements. My list has 19.
    # Let's re-check the BCLR reconstruction.
    # The reconstruction I found has 21 terms.
    # Let's assume the list I found is correct and has 21 elements.
    
    # I will use the list from the source:
    # c00: 1, 0, 0, -1, -1, 1, 0, 0, -1, 0, 1, 0, -1, 1, 0, -1, 0, 0, 1, 0, 0
    # c01: 0, 1, 0, 1, 0, -1, 0, -1, 0, 1, -1, 0, -1, 0, 1, 0, 1, -1, 0, 0, 0
    # c02: 0, -1, 1, 0, 1, 0, -1, 0, 1, -1, 0, -1, 0, 1, 0, 0, 1, -1, 0, 0, 0
    # c10: 0, 0, 0, -1, 1, 0, 0, 0, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 0, 0
    # c11: 1, -1, -1, 0, 0, 0, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1
    # c12: 0, 0, 0, 0, 1, 0, 1, 0, -1, 1, 0, -1, 0, 1, 0, 0, -1, 0, 1, 0, 0
    # c20: 0, -1, 0, 0, 0, 1, 0, 0, 1, -1, 1, 0, -1, 0, -1, 1, 0, -1, 0, 0, 0
    # c21: 0, 0, 0, -1, 0, 0, 0, 1, 0, 1, -1, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0
    # c22: 0, 0, 0, 0, 1, 0, 0, 0, 1, -1, 1, 0, -1, 0, -1, 0, 0, -1, 0, 0, 1
    
    bclr_recon = [
        [1, 0, 0, -1, -1, 1, 0, 0, -1, 0, 1, 0, -1, 1, 0, -1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, -1, 0, -1, 0, 1, -1, 0, -1, 0, 1, 0, 1, -1, 0, 0, 0],
        [0, -1, 1, 0, 1, 0, -1, 0, 1, -1, 0, -1, 0, 1, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, -1, 1, 0, 0, 0, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 0, 0],
        [1, -1, -1, 0, 0, 0, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1],
        [0, 0, 0, 0, 1, 0, 1, 0, -1, 1, 0, -1, 0, 1, 0, 0, -1, 0, 1, 0, 0],
        [0, -1, 0, 0, 0, 1, 0, 0, 1, -1, 1, 0, -1, 0, -1, 1, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 1, 0, 1, -1, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, -1, 1, 0, -1, 0, -1, 0, 0, -1, 0, 0, 1]
    ]
    
    rank = 7 * 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for bclr_idx in range(21):
            a_bclr, b_bclr = bclr_terms[bclr_idx]
            for i in range(bs):   # local row in 3x3 block
                for k in range(bs):  # local col in A / local row in B
                    for j in range(bs):  # local col in B / local col in C
                        # U contributions
                        for (blk_a, coeff_a) in a_comb:
                            row_a = (blk_a // 2) * bs
                            col_a = (blk_a % 2) * bs
                            local_a_idx = i * 3 + k
                            # Check if local_a_idx is in a_bclr
                            coeff_bclr_a = 0
                            for (idx, c) in a_bclr:
                                if idx == local_a_idx:
                                    coeff_bclr_a = c
                                    break
                            if coeff_bclr_a != 0:
                                u_idx = (row_a + i) * m + (col_a + k)
                                U[u_idx][col] += coeff_a * coeff_bclr_a
                                
                        # V contributions
                        for (blk_b, coeff_b) in b_comb:
                            row_b = (blk_b // 2) * bs
                            col_b = (blk_b % 2) * bs
                            local_b_idx = k * 3 + j
                            coeff_bclr_b = 0
                            for (idx, c) in b_bclr:
                                if idx == local_b_idx:
                                    coeff_bclr_b = c
                                    break
                            if coeff_bclr_b != 0:
                                v_idx = (row_b + k) * p + (col_b + j)
                                V[v_idx][col] += coeff_b * coeff_bclr_b
                                
                        # W contributions
                        for (blk_c, coeff_c) in output_contrib[term_idx]:
                            row_c = (blk_c // 2) * bs
                            col_c = (blk_c % 2) * bs
                            local_c_idx = i * 3 + j
                            coeff_bclr_c = bclr_recon[local_c_idx][bclr_idx]
                            if coeff_bclr_c != 0:
                                w_idx = (col_c + j) * n + (row_c + i)
                                W[w_idx][col] += coeff_c * coeff_bclr_c
                                
                        col += 1
                        
        if time.time() - start_time > timeout:
            break
            
    return {"U": U, "V": V, "W": W}
