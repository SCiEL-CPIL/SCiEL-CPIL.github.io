"""Improved recursive Strassen for 6x6x6 matrix multiplication.
Uses 2-level recursive Strassen with careful handling of block sizes.
Level 1: 6x6 = 2x2 blocks of 3x3 (7 Strassen terms)
Level 2: 3x3 handled via rank-27 schoolbook decomposition.
Total rank = 7 * 27 = 189.
"""

import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit
    
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Level 1 Strassen: 6x6 split into 2x2 blocks of 3x3
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # Each block is 3x3
    
    # Strassen terms: (left_combo, right_combo)
    # Each combo maps block_index -> coefficient
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_l1 = [
        # T0: (A00+A11)*(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # T1: (A10+A11)*B00
        ({2: 1, 3: 1}, {0: 1}),
        # T2: A00*(B01-B11)
        ({0: 1}, {1: 1, 3: -1}),
        # T3: A11*(B10-B00)
        ({3: 1}, {2: 1, 0: -1}),
        # T4: (A00+A01)*B11
        ({0: 1, 1: 1}, {3: 1}),
        # T5: (A10-A00)*(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # T6: (A01-A11)*(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # Result contributions: C_block = sum of signed Tj
    # C00 = T0 + T3 - T4 + T6
    # C01 = T2 + T4
    # C10 = T1 + T3
    # C11 = T0 - T1 + T2 + T5
    result_l1 = {
        0: [(0, 1), (3, 1), (4, -1), (6, 1)],
        1: [(2, 1), (4, 1)],
        2: [(1, 1), (3, 1)],
        3: [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    # For each Strassen term, decompose into 27 schoolbook scalar multiplications
    # for the 3x3 product
    
    col_idx = 0
    for t in range(7):
        if time.time() - start_time > DEADLINE:
            # Return what we have so far (partial but valid structure if complete)
            # In practice, this shouldn't trigger since the loop is fast
            pass
        
        left_combo, right_combo = strassen_l1[t]
        
        # For each schoolbook multiplication (i, a, k) in 0..2:
        # X[i,a] * Y[a,k] where:
        # X[i,a] = sum_{b in left_combo} coeff * A[block_b row i, col a]
        # Y[a,k] = sum_{b in right_combo} coeff * B[block_b row a, col k]
        # Contribution to C at positions determined by result_l1
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # Fill U: coefficient for each A entry
                    for blk, coeff_l in left_combo.items():
                        br, bc = divmod(blk, 2)
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * m + col_A][col_idx] += coeff_l
                    
                    # Fill V: coefficient for each B entry
                    for blk, coeff_r in right_combo.items():
                        br, bc = divmod(blk, 2)
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * p + col_B][col_idx] += coeff_r
                    
                    # Fill W: contribution to C
                    for cblk, signs in result_l1.items():
                        for (si, sign) in signs:
                            if si == t:
                                rbr, rbc = divmod(cblk, 2)
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                W_row = k_total * n + i_total
                                W[W_row][col_idx] += sign
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
