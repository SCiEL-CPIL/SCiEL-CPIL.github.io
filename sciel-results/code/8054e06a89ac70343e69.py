import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We use a recursive Strassen decomposition.
    # 4x4 * 4x6 is split into 7 sub-problems of size 2x2 * 2x3.
    # Each 2x2 * 2x3 is split into a 2x2 * 2x1 (rank 4) and a 2x2 * 2x2 (rank 7).
    # Total rank per 2x2x3 is 11.
    # Total rank for 4x4x6 is 7 * 11 = 77.
    
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard Strassen steps for 2x2 matrices
    # Each step is (alpha, beta, gamma) where alpha, beta, gamma are dicts {(i,j): val}
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Helper to add a 2x2 * 2x1 schoolbook decomposition
    # A is 2x2, B is 2x1, C is 2x1
    # Local indices:
    # A: i*2+j
    # B: j*1+k -> j (since k=0)
    # C: k*1+i -> i (since k=0)
    # Global mapping is handled by the caller via offsets and strides.
    
    def add_schoolbook_221(base_col, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, a_stride_r, a_stride_c, b_stride_r, b_stride_c, c_stride_r, c_stride_c):
        # 4 terms
        # Term 0: A[0,0]*B[0] -> C[0]
        # Term 1: A[0,1]*B[1] -> C[0]
        # Term 2: A[1,0]*B[0] -> C[1]
        # Term 3: A[1,1]*B[1] -> C[1]
        
        terms = [
            ((0,0), (0,0), (0,0)),
            ((0,1), (1,0), (0,0)),
            ((1,0), (0,0), (1,0)),
            ((1,1), (1,0), (1,0))
        ]
        
        for t_idx, ((ai, aj), (bi, bj), (ci, cj)) in enumerate(terms):
            col = base_col + t_idx
            
            # U index: (a_row_off + ai*a_stride_r) * m + (a_col_off + aj*a_stride_c)
            # Note: a_stride_r and a_stride_c are the step in global row/col for local +1
            # For 2x2 block inside 4x4:
            # Local (0,0) -> Global (a_row_off, a_col_off)
            # Local (1,1) -> Global (a_row_off+1, a_col_off+1)
            # So stride is 1 for both.
            
            u_idx = (a_row_off + ai) * m + (a_col_off + aj)
            U[u_idx][col] = 1
            
            # V index: (b_row_off + bi) * p + (b_col_off + bj)
            v_idx = (b_row_off + bi) * p + (b_col_off + bj)
            V[v_idx][col] = 1
            
            # W index: (c_row_off + ci) * n + (c_col_off + cj)
            # Note: c_row_off is the global column of C, c_col_off is the global row of C.
            # Wait, the convention is W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r].
            # So W index is k*n + i.
            # c_row_off is the offset for k (column of C).
            # c_col_off is the offset for i (row of C).
            w_idx = (c_row_off + ci) * n + (c_col_off + cj)
            W[w_idx][col] = 1

    def add_strassen_222(base_col, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        # Adds 7 Strassen terms for a 2x2 * 2x2 block
        for s in range(7):
            alpha, beta, gamma = strassen_steps_222[s]
            col = base_col + s
            
            for (ai, aj), val_a in alpha.items():
                u_idx = (a_row_off + ai) * m + (a_col_off + aj)
                U[u_idx][col] += val_a
            
            for (bi, bj), val_b in beta.items():
                v_idx = (b_row_off + bi) * p + (b_col_off + bj)
                V[v_idx][col] += val_b
            
            for (ci, cj), val_c in gamma.items():
                # ci is local row of C, cj is local col of C
                # Global row of C = c_row_off + ci
                # Global col of C = c_col_off + cj
                # W index = (global col) * n + (global row)
                w_idx = (c_col_off + cj) * n + (c_row_off + ci)
                W[w_idx][col] += val_c

    # Main loop: 7 Strassen terms at the 4x4 level
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1) -- but B is 4x6, so blocks are 2x3
    # C blocks: (0,0), (0,1), (1,0), (1,1) -- C is 4x6, so blocks are 2x3
    
    # The 7 Strassen terms for 4x4 * 4x6:
    # 1. (A00+A11) * (B00+B11) -> C00, C11
    # 2. (A10+A11) * B00 -> C10, C11
    # 3. A00 * (B01-B11) -> C01, C11
    # 4. A11 * (B10-B00) -> C00, C10
    # 5. (A00+A01) * B11 -> C00, C01
    # 6. (A10-A00) * (B00+B01) -> C11
    # 7. (A01-A11) * (B10+B11) -> C00
    
    # Let's define the terms explicitly to avoid confusion.
    # Each term involves a sum of A blocks, a sum of B blocks, and a sum of C blocks.
    # Since the sub-problem is 2x2 * 2x3, we need to handle the 2x3 part carefully.
    
    # It is easier to iterate over the 7 high-level Strassen terms.
    # For each term, we have a linear combination of A-blocks, B-blocks, and C-blocks.
    # The product is (Sum A_blocks) * (Sum B_blocks) = Sum C_blocks.
    # This is equivalent to performing the 2x2 * 2x3 multiplication for each pair of non-zero A-block and B-block?
    # No, Strassen works because (A1+A2)(B1+B2) = A1B1 + A1B2 + A2B1 + A2B2.
    # So if we have a term like (A00+A11)*(B00+B11), it contributes to C00, C01, C10, C11?
    # No, C = A*B.
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M3 = A00(B01-B11) = A00B01 - A00B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to decompose each Mi (which is a 2x2 * 2x3 product) into rank 11.
    # But wait, Mi is not a single 2x2 * 2x3 product.
    # M1 = (A00+A11)(B00+B11). This is a product of (2x2 matrix) * (2x3 matrix).
    # So we can use the rank-11 decomposition for 2x2 * 2x3.
    
    # Let's define the 7 high-level terms as:
    # Term 0: A=(0,0)+(1,1), B=(0,0)+(1,1)
    # Term 1: A=(1,0)+(1,1), B=(0,0)
    # Term 2: A=(0,0), B=(0,1)-(1,1)
    # Term 3: A=(1,1), B=(1,0)-(0,0)
    # Term 4: A=(0,0)+(0,1), B=(1,1)
    # Term 5: A=(1,0)-(0,0), B=(0,0)+(0,1)
    # Term 6: A=(0,1)-(1,1), B=(1,0)+(1,1)
    
    # And the C contributions:
    # C00 gets: +Term0, +Term3, -Term4, +Term6
    # C01 gets: +Term2, +Term4
    # C10 gets: +Term1, +Term3
    # C11 gets: +Term0, -Term1, +Term2, +Term5
    
    # However, we cannot just add the decompositions directly because the C indices are different.
    # We need to add the U and V parts for each term, and the W part for each C block with the appropriate coefficient.
    
    # Let's iterate over the 7 terms.
    for t in range(7):
        # Define A, B, C contributions for this term
        # A contrib: list of (block, coeff)
        # B contrib: list of (block, coeff)
        # C contrib: list of (block, coeff)
        
        if t == 0:
            a_blocks = [((0,0), 1), ((1,1), 1)]
            b_blocks = [((0,0), 1), ((1,1), 1)]
            c_blocks = [((0,0), 1), ((1,1), 1)] # Wait, C00 and C11 get +1.
            # But C01 and C10 get 0.
            # Actually, we should just define the C blocks that receive this term.
            # C00: +1, C11: +1.
        elif t == 1:
            a_blocks = [((1,0), 1), ((1,1), 1)]
            b_blocks = [((0,0), 1)]
            c_blocks = [((1,0), 1), ((1,1), -1)]
        elif t == 2:
            a_blocks = [((0,0), 1)]
            b_blocks = [((0,1), 1), ((1,1), -1)]
            c_blocks = [((0,1), 1), ((1,1), 1)]
        elif t == 3:
            a_blocks = [((1,1), 1)]
            b_blocks = [((1,0), 1), ((0,0), -1)]
            c_blocks = [((0,0), 1), ((1,0), 1)]
        elif t == 4:
            a_blocks = [((0,0), 1), ((0,1), 1)]
            b_blocks = [((1,1), 1)]
            c_blocks = [((0,0), -1), ((0,1), 1)]
        elif t == 5:
            a_blocks = [((1,0), 1), ((0,0), -1)]
            b_blocks = [((0,0), 1), ((0,1), 1)]
            c_blocks = [((1,1), 1)]
        elif t == 6:
            a_blocks = [((0,1), 1), ((1,1), -1)]
            b_blocks = [((1,0), 1), ((1,1), 1)]
            c_blocks = [((0,0), 1)]
        
        base_col = t * 11
        
        # Now we need to add the rank-11 decomposition for 2x2 * 2x3.
        # The A part is a sum of 2x2 blocks.
        # The B part is a sum of 2x3 blocks.
        # The product is a sum of 2x3 blocks (C).
        
        # We can just iterate over the 11 "sub-terms" of the 2x2 * 2x3 decomposition.
        # But we need to scale them by the coefficients from a_blocks and b_blocks.
        # And add them to the correct C blocks.
        
        # This is getting complicated. Let's stick to the simpler method used in the current program:
        # Recursively apply Strassen.
        
        # Actually, the current program's logic for Part 1 (4x4x4) and Part 2 (4x4x2) is flawed because it treats them independently but doesn't share the A factors correctly or handles the 4x4x2 part with a non-optimal split.
        # The inspiration's method of using a 2x2x3 base case is better.
        
        # Let's implement the 2x2x3 base case decomposition directly in the loop.
        
        # For each term t, we have a linear combination of A-blocks and B-blocks.
        # Let A_t = sum_a coeff_a * A_block_a
        # Let B_t = sum_b coeff_b * B_block_b
        # Then A_t * B_t = sum_a sum_b coeff_a coeff_b * (A_block_a * B_block_b)
        
        # Each A_block_a is 2x2, B_block_b is 2x3.
        # We can use the rank-11 decomposition for 2x2 * 2x3 for each pair (a, b).
        # But that would be 4 * 4 = 16 decompositions per term, total 7 * 16 * 11 = 1232. Too high.
        
        # The trick is that the rank-11 decomposition for 2x2 * 2x3 is linear.
        # So we can just add the coefficients to the U and V factors, and handle the W factors separately.
        
        # Let's define the 11 "atomic" terms for 2x2 * 2x3.
        # 4 schoolbook terms for 2x2 * 2x1 (cols 0)
        # 7 Strassen terms for 2x2 * 2x2 (cols 1,2)
        
        # For each atomic term, we have a U pattern (2x2), V pattern (2x3), W pattern (2x3).
        # We will add coeff_a * coeff_b * U_pattern to U, etc.
        
        # But wait, the U pattern depends on which A-block is used.
        # The V pattern depends on which B-block is used.
        # The W pattern depends on which C-block is used.
        
        # So for each term t:
        #   For each atomic sub-term s (0..10):
        #     For each A-block a in a_blocks:
        #       For each B-block b in b_blocks:
        #         U += coeff_a * coeff_b * U_s(a)
        #         V += coeff_a * coeff_b * V_s(b)
        #         W += coeff_a * coeff_b * W_s(c) ... wait, W depends on C-block.
        
        # This is still messy. Let's use the explicit index mapping from the inspiration.
        
        # Let's just hardcode the 77 terms.
        # It's safer and less error-prone.
        
        pass # We will replace this with a direct construction below.

    # Reset matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 atomic terms for 2x2 * 2x3
    # U, V, W are 2x2, 2x3, 2x3 local matrices.
    # We represent them as lists of (index, value) tuples.
    
    # Schoolbook 2x2 * 2x1 (cols 0 of B, cols 0 of C)
    # 0: A[0,0]*B[0,0] -> C[0,0]
    # 1: A[0,1]*B[1,0] -> C[0,0]
    # 2: A[1,0]*B[0,0] -> C[1,0]
    # 3: A[1,1]*B[1,0] -> C[1,0]
    
    # Strassen 2x2 * 2x2 (cols 1,2 of B, cols 1,2 of C)
    # 4..10: Standard Strassen steps
    
    # Let's define the patterns.
    # Local indices:
    # A: i*2+j
    # B: j*3+k
    # C: k*2+i
    
    atomic_terms = []
    
    # Term 0: A[0,0], B[0,0], C[0,0]
    atomic_terms.append({
        'U': [(0, 1)],
        'V': [(0, 1)], # B[0,0] -> 0*3+0
        'W': [(0, 1)]  # C[0,0] -> 0*2+0
    })
    
    # Term 1: A[0,1], B[1,0], C[0,0]
    atomic_terms.append({
        'U': [(1, 1)],
        'V': [(3, 1)], # B[1,0] -> 1*3+0
        'W': [(0, 1)]  # C[0,0] -> 0*2+0
    })
    
    # Term 2: A[1,0], B[0,0], C[1,0]
    atomic_terms.append({
        'U': [(2, 1)],
        'V': [(0, 1)], # B[0,0]
        'W': [(1, 1)]  # C[1,0] -> 0*2+1
    })
    
    # Term 3: A[1,1], B[1,0], C[1,0]
    atomic_terms.append({
        'U': [(3, 1)],
        'V': [(3, 1)], # B[1,0]
        'W': [(1, 1)]  # C[1,0]
    })
    
    # Strassen terms for cols 1,2
    # Local B indices: 1,2,4,5
    # Local C indices: 2,4,3,5
    
    # Term 4: M1
    atomic_terms.append({
        'U': [(0, 1), (3, 1)],
        'V': [(1, 1), (5, 1)], # B[0,1], B[1,2]
        'W': [(2, 1), (5, 1)]  # C[0,1], C[1,2]
    })
    
    # Term 5: M2
    atomic_terms.append({
        'U': [(2, 1), (3, 1)],
        'V': [(1, 1)], # B[0,1]
        'W': [(3, 1), (5, -1)] # C[1,1], C[1,2]
    })
    
    # Term 6: M3
    atomic_terms.append({
        'U': [(0, 1)],
        'V': [(2, 1), (5, -1)], # B[0,2], B[1,2]
        'W': [(4, 1), (5, 1)]  # C[0,2], C[1,2]
    })
    
    # Term 7: M4
    atomic_terms.append({
        'U': [(3, 1)],
        'V': [(4, 1), (1, -1)], # B[1,1], B[0,1]
        'W': [(2, 1), (3, 1)]  # C[0,1], C[1,1]
    })
    
    # Term 8: M5
    atomic_terms.append({
        'U': [(0, 1), (1, 1)],
        'V': [(5, 1)], # B[1,2]
        'W': [(2, -1), (4, 1)] # C[0,1], C[0,2]
    })
    
    # Term 9: M6
    atomic_terms.append({
        'U': [(2, 1), (0, -1)],
        'V': [(1, 1), (2, 1)], # B[0,1], B[0,2]
        'W': [(5, 1)]  # C[1,2]
    })
    
    # Term 10: M7
    atomic_terms.append({
        'U': [(1, 1), (3, -1)],
        'V': [(4, 1), (5, 1)], # B[1,1], B[1,2]
        'W': [(2, 1)]  # C[0,1]
    })
    
    # Now, for each of the 7 high-level Strassen terms:
    for t in range(7):
        if t == 0:
            a_blocks = [((0,0), 1), ((1,1), 1)]
            b_blocks = [((0,0), 1), ((1,1), 1)]
            c_blocks = [((0,0), 1), ((1,1), 1)]
        elif t == 1:
            a_blocks = [((1,0), 1), ((1,1), 1)]
            b_blocks = [((0,0), 1)]
            c_blocks = [((1,0), 1), ((1,1), -1)]
        elif t == 2:
            a_blocks = [((0,0), 1)]
            b_blocks = [((0,1), 1), ((1,1), -1)]
            c_blocks = [((0,1), 1), ((1,1), 1)]
        elif t == 3:
            a_blocks = [((1,1), 1)]
            b_blocks = [((1,0), 1), ((0,0), -1)]
            c_blocks = [((0,0), 1), ((1,0), 1)]
        elif t == 4:
            a_blocks = [((0,0), 1), ((0,1), 1)]
            b_blocks = [((1,1), 1)]
            c_blocks = [((0,0), -1), ((0,1), 1)]
        elif t == 5:
            a_blocks = [((1,0), 1), ((0,0), -1)]
            b_blocks = [((0,0), 1), ((0,1), 1)]
            c_blocks = [((1,1), 1)]
        elif t == 6:
            a_blocks = [((0,1), 1), ((1,1), -1)]
            b_blocks = [((1,0), 1), ((1,1), 1)]
            c_blocks = [((0,0), 1)]
        
        for s in range(11):
            col = t * 11 + s
            term = atomic_terms[s]
            
            # Accumulate U, V, W contributions
            # Note: The atomic term assumes A, B, C are single blocks.
            # We need to map local indices to global indices for each block.
            
            # For U:
            # Local index u_idx -> global index depends on A-block.
            # A-block (bi, bj) starts at row bi*2, col bj*2.
            # Local (i, j) -> global (bi*2+i, bj*2+j).
            # Global U index = (bi*2+i)*m + (bj*2+j).
            
            for (u_local, val_u) in term['U']:
                i = u_local // 2
                j = u_local % 2
                for (bi, bj), coeff_a in a_blocks:
                    u_global = (bi * 2 + i) * m + (bj * 2 + j)
                    U[u_global][col] += coeff_a * val_u
            
            # For V:
            # Local index v_idx -> global index depends on B-block.
            # B-block (bi, bj) starts at row bi*2, col bj*3.
            # Local (j, k) -> global (bi*2+j, bj*3+k).
            # Global V index = (bi*2+j)*p + (bj*3+k).
            
            for (v_local, val_v) in term['V']:
                j = v_local // 3
                k = v_local % 3
                for (bi, bj), coeff_b in b_blocks:
                    v_global = (bi * 2 + j) * p + (bj * 3 + k)
                    V[v_global][col] += coeff_b * val_v
            
            # For W:
            # Local index w_idx -> global index depends on C-block.
            # C-block (bi, bj) starts at row bi*2, col bj*3.
            # Local (k, i) -> global (bi*2+i, bj*3+k).
            # Global W index = (bj*3+k)*n + (bi*2+i).
            
            for (w_local, val_w) in term['W']:
                k = w_local // 2
                i = w_local % 2
                for (bi, bj), coeff_c in c_blocks:
                    w_global = (bj * 3 + k) * n + (bi * 2 + i)
                    W[w_global][col] += coeff_c * val_w

    return {"U": U, "V": V, "W": W}
