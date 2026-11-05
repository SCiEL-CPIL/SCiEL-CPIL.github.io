"""
Matrix multiplication 4x4 * 4x5 decomposition.

Strategy: Use a recursive Strassen-based approach.
1. Split the 4x4 * 4x5 multiplication into four 2x2 block multiplications 
   (treating the 5 columns as 2+3 or similar, but standard Strassen splits 4x4 into 2x2).
   Actually, standard Strassen splits n x n into n/2 x n/2.
   For 4x4 * 4x5, we can pad B to 4x6 or 4x8, or just handle the blocks carefully.
   
   Let's stick to a clean decomposition:
   Split A (4x4) into 2x2 blocks: A00, A01, A10, A11.
   Split B (4x5) into 2x2 blocks? No, B is 4x5.
   We can split B into B_left (4x2) and B_right (4x3)?
   Or pad B to 4x4 (zero padding the last column) and handle the 5th column separately?
   
   Let's try:
   C = A * B
   Split A into 2x2 blocks of size 2x2.
   Split B into 2x2 blocks? B is 4x5.
   Let B = [B0, B1] where B0 is 4x2 and B1 is 4x3.
   Then C = [A*B0, A*B1].
   
   A*B0: 4x4 * 4x2 -> 4x2.
   A*B1: 4x4 * 4x3 -> 4x3.
   
   This is asymmetric.
   
   Alternative: Pad B to 4x4 with a zero column B4.
   B_padded = [B0, B1, B2, B3, 0] is not square.
   Let's pad B to 4x4 by ignoring the 5th column? No, we need the 5th column.
   
   Let's pad B to 4x4 by taking the first 4 columns, and treat the 5th column separately?
   B = [B_4x4, B_4x1].
   C = A * B_4x4 + A * B_4x1.
   
   Part 1: A (4x4) * B_4x4 (4x4).
   Use Strassen for 4x4 * 4x4.
   Strassen for 4x4 splits into 2x2 blocks.
   A = [[A00, A01], [A10, A11]] where each is 2x2.
   B_4x4 = [[B00, B01], [B10, B11]] where each is 2x2.
   
   Strassen computes 7 products of 2x2 matrices.
   Each 2x2 product can be computed by:
   1. Schoolbook (4 mults)
   2. Strassen (7 mults)
   
   If we use Strassen recursively:
   Rank for 4x4 * 4x4 = 7 * (Rank for 2x2 * 2x2).
   Rank for 2x2 * 2x2 using Strassen = 7.
   So Rank for 4x4 * 4x4 = 7 * 7 = 49.
   
   Part 2: A (4x4) * B_4x1 (4x1).
   This is a matrix-vector product.
   C_vec[i] = sum_j A[i,j] * B[j,0].
   This is 4x4 = 16 multiplications.
   Can we do better?
   The tensor for 4x4 * 4x1 has rank 16?
   T[i,j,k=0] = delta_{i,?}... no.
   C[i] = sum_j A[i,j] B[j,0].
   The tensor is T[i,j,0] = 1? No, T is 3D.
   Indices: i (0..3), j (0..3), k (0).
   T[i,j,k] = 1 if k=0? No.
   C[i] = sum_j A[i,j] B[j,0].
   The term is A[i,j] * B[j,0].
   This is a sum of 16 rank-1 terms: sum_{i,j} A[i,j] B[j,0] * (contribution to C[i]).
   The contribution to C[i] is 1.
   So U[i*m+j] = 1, V[j*p+0] = 1, W[0*n+i] = 1.
   Rank is 16.
   
   Total Rank = 49 + 16 = 65.
   
   This matches the inspiration (score 1.2308).
   
   Let's implement this.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will use a decomposition of rank 65.
    # 49 terms for 4x4 * 4x4 (Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
    # We compute C_4x4 = A * B_4x4
    # A is 4x4, B_4x4 is the first 4 columns of B.
    
    # Strassen decomposition for 4x4 * 4x4
    # Splits into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # The 7 Strassen products are 2x2 matrix multiplications.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 product. We compute each using Strassen (7 scalar mults).
    # Total 49 scalar mults.
    
    # Helper to expand a 2x2 matrix product P*Q into scalar mults
    # and map them to U, V, W indices.
    # P is a linear combination of A blocks.
    # Q is a linear combination of B blocks.
    
    # We need to define the linear combinations for P and Q for each of the 7 Strassen terms.
    # Then for each term, we expand the 2x2 product.
    
    # Let's define the block coefficients for each Strassen term.
    # P_blocks: [coeff_A00, coeff_A01, coeff_A10, coeff_A11]
    # Q_blocks: [coeff_B00, coeff_B01, coeff_B10, coeff_B11]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1])  # M7
    ]
    
    # For each term, we compute the 2x2 product.
    # The result of each 2x2 product is added to the final C with specific coefficients.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to add the contribution of each scalar mult in the 2x2 product
    # to the correct block of C.
    
    # Let's iterate over the 7 Strassen terms.
    for t_idx, (p_blocks, q_blocks) in enumerate(strassen_terms):
        # For this term, we compute P*Q where P and Q are 2x2 matrices.
        # P is formed by linear combinations of A blocks.
        # Q is formed by linear combinations of B blocks.
        
        # We will generate 7 scalar mults for this 2x2 product using Strassen.
        # Or just schoolbook (4 mults) for simplicity?
        # Strassen for 2x2 is 7 mults.
        # Let's use Strassen for the inner 2x2 product to get rank 7 per term.
        
        # Inner Strassen terms for P*Q:
        # s1 = (P00+P11)(Q00+Q11)
        # s2 = (P10+P11)Q00
        # s3 = P00(Q01-Q11)
        # s4 = P11(Q10-Q00)
        # s5 = (P00+P01)Q11
        # s6 = (P10-P00)(Q00+Q01)
        # s7 = (P01-P11)(Q10+Q11)
        
        # Inner Strassen coefficients:
        inner_strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1]),  # s1
            ([0, 0, 1, 1], [1, 0, 0, 0]),  # s2
            ([1, 0, 0, 0], [0, 1, 0, -1]), # s3
            ([0, 0, 0, 1], [-1, 0, 1, 0]), # s4
            ([1, 1, 0, 0], [0, 0, 0, 1]),  # s5
            ([-1, 0, 1, 0], [1, 1, 0, 0]), # s6
            ([0, 1, 0, -1], [0, 0, 1, 1])  # s7
        ]
        
        # For each inner term, we have a scalar multiplication.
        # We need to map this to U, V, W.
        
        # The scalar mult is: (sum p_ij A_block_ij) * (sum q_kl B_block_kl)
        # where p_ij are the coefficients for the 2x2 P matrix elements.
        # Wait, the inner Strassen terms operate on the elements of P and Q.
        # P is 2x2. P00, P01, P10, P11.
        # Q is 2x2. Q00, Q01, Q10, Q11.
        
        # The inner Strassen terms are linear combinations of P elements and Q elements.
        # We need to expand these linear combinations in terms of A and B blocks.
        
        for s_idx, (p_elem_coeffs, q_elem_coeffs) in enumerate(inner_strassen_terms):
            # p_elem_coeffs: [coeff_P00, coeff_P01, coeff_P10, coeff_P11]
            # q_elem_coeffs: [coeff_Q00, coeff_Q01, coeff_Q10, coeff_Q11]
            
            # Build u_list, v_list, w_list for this scalar mult.
            u_list = []
            v_list = []
            w_list = []
            
            # Expand P elements in terms of A blocks.
            # P is a 2x2 matrix formed from A blocks.
            # P[r, c] = sum_{b} p_blocks[b] * A_block[b][r, c]
            # where b is 0..3 (A00, A01, A10, A11).
            # A_block[b] is a 2x2 matrix.
            
            # We need to identify the global indices (i, j) for A[i, j].
            # A_block[0] (A00): i in 0..1, j in 0..1.
            # A_block[1] (A01): i in 0..1, j in 2..3.
            # A_block[2] (A10): i in 2..3, j in 0..1.
            # A_block[3] (A11): i in 2..3, j in 2..3.
            
            # For each element (r, c) of P (r, c in 0..1):
            # P[r, c] = sum_{b} p_blocks[b] * A_block[b][r, c]
            
            # So the coefficient of A[i, j] in the linear combination for P_elem is:
            # For P_elem = r*2 + c:
            # coeff_A[i, j] = sum_{b} p_elem_coeffs[r*2+c] * p_blocks[b] * delta_{i,j in block b}
            
            # Let's compute the coefficients for A.
            # We iterate over r, c (0..1) and b (0..3).
            for r in range(2):
                for c in range(2):
                    p_elem_idx = r * 2 + c
                    coeff_p_elem = p_elem_coeffs[p_elem_idx]
                    if coeff_p_elem == 0:
                        continue
                    
                    for b in range(4):
                        coeff_p_block = p_blocks[b]
                        if coeff_p_block == 0:
                            continue
                        
                        # Determine global i, j for A_block[b][r, c]
                        if b == 0: # A00
                            gi = r
                            gj = c
                        elif b == 1: # A01
                            gi = r
                            gj = 2 + c
                        elif b == 2: # A10
                            gi = 2 + r
                            gj = c
                        elif b == 3: # A11
                            gi = 2 + r
                            gj = 2 + c
                        else:
                            continue
                        
                        coeff_a = coeff_p_elem * coeff_p_block
                        if coeff_a != 0:
                            u_list.append((gi, gj, coeff_a))
            
            # Similarly for Q and B.
            # Q[r, c] = sum_{b} q_blocks[b] * B_block[b][r, c]
            # B_block[0] (B00): j in 0..1, k in 0..1.
            # B_block[1] (B01): j in 0..1, k in 2..3.
            # B_block[2] (B10): j in 2..3, k in 0..1.
            # B_block[3] (B11): j in 2..3, k in 2..3.
            
            for r in range(2):
                for c in range(2):
                    q_elem_idx = r * 2 + c
                    coeff_q_elem = q_elem_coeffs[q_elem_idx]
                    if coeff_q_elem == 0:
                        continue
                    
                    for b in range(4):
                        coeff_q_block = q_blocks[b]
                        if coeff_q_block == 0:
                            continue
                        
                        # Determine global j, k for B_block[b][r, c]
                        if b == 0: # B00
                            gj = r
                            gk = c
                        elif b == 1: # B01
                            gj = r
                            gk = 2 + c
                        elif b == 2: # B10
                            gj = 2 + r
                            gk = c
                        elif b == 3: # B11
                            gj = 2 + r
                            gk = 2 + c
                        else:
                            continue
                        
                        coeff_b = coeff_q_elem * coeff_q_block
                        if coeff_b != 0:
                            v_list.append((gj, gk, coeff_b))
            
            # Now we need to determine the contribution to C.
            # The scalar mult is s_idx of the inner Strassen for term t_idx.
            # We need to know how s_idx contributes to the 2x2 result of P*Q.
            # Inner Strassen result:
            # C00 = s1 + s4 - s5 + s7
            # C01 = s3 + s5
            # C10 = s2 + s4
            # C11 = s1 - s2 + s3 + s6
            
            # Let's define the coefficients for each s_idx in the 2x2 result.
            # result_coeffs[local_r, local_c] for s_idx.
            
            if s_idx == 0: # s1
                res_coeffs = {(0,0): 1, (1,1): 1}
            elif s_idx == 1: # s2
                res_coeffs = {(1,0): 1, (1,1): -1}
            elif s_idx == 2: # s3
                res_coeffs = {(0,1): 1, (1,1): 1}
            elif s_idx == 3: # s4
                res_coeffs = {(0,0): 1, (1,0): 1}
            elif s_idx == 4: # s5
                res_coeffs = {(0,0): -1, (0,1): 1}
            elif s_idx == 5: # s6
                res_coeffs = {(1,1): 1}
            elif s_idx == 6: # s7
                res_coeffs = {(0,0): 1}
            
            # Now, the 2x2 result of P*Q is added to the final C (4x4) based on t_idx.
            # t_idx 0 (M1): C00 += M1, C11 += M1
            # t_idx 1 (M2): C10 += M2, C11 -= M2
            # t_idx 2 (M3): C01 += M3, C11 += M3
            # t_idx 3 (M4): C00 += M4, C10 += M4
            # t_idx 4 (M5): C00 -= M5, C01 += M5
            # t_idx 5 (M6): C11 += M6
            # t_idx 6 (M7): C00 += M7
            
            # C00: i in 0..1, k in 0..1
            # C01: i in 0..1, k in 2..3
            # C10: i in 2..3, k in 0..1
            # C11: i in 2..3, k in 2..3
            
            # For each (local_r, local_c, coeff) in res_coeffs:
            # The contribution is coeff * (scalar mult).
            # This adds to C_block[local_r, local_c].
            # We need to map (local_r, local_c) to global (i, k) for C.
            
            # C_block mapping for t_idx:
            # t_idx 0: C00 (i=0..1, k=0..1) and C11 (i=2..3, k=2..3)
            # t_idx 1: C10 (i=2..3, k=0..1) and C11 (i=2..3, k=2..3) with sign -1
            # t_idx 2: C01 (i=0..1, k=2..3) and C11 (i=2..3, k=2..3)
            # t_idx 3: C00 (i=0..1, k=0..1) and C10 (i=2..3, k=0..1)
            # t_idx 4: C00 (i=0..1, k=0..1) with sign -1 and C01 (i=0..1, k=2..3)
            # t_idx 5: C11 (i=2..3, k=2..3)
            # t_idx 6: C00 (i=0..1, k=0..1)
            
            # Let's define the target blocks and signs for each t_idx.
            if t_idx == 0:
                targets = [(0, 1), (2, 3)] # (i_start, k_start) for C00, C11
                signs = [1, 1]
            elif t_idx == 1:
                targets = [(2, 0), (2, 2)] # C10, C11
                signs = [1, -1]
            elif t_idx == 2:
                targets = [(0, 2), (2, 2)] # C01, C11
                signs = [1, 1]
            elif t_idx == 3:
                targets = [(0, 0), (2, 0)] # C00, C10
                signs = [1, 1]
            elif t_idx == 4:
                targets = [(0, 0), (0, 2)] # C00, C01
                signs = [-1, 1]
            elif t_idx == 5:
                targets = [(2, 2)] # C11
                signs = [1]
            elif t_idx == 6:
                targets = [(0, 0)] # C00
                signs = [1]
            
            for target_idx, (i_start, k_start) in enumerate(targets):
                sign = signs[target_idx]
                for (lr, lc), coeff_res in res_coeffs.items():
                    # lr, lc are local indices in the 2x2 result (0..1).
                    # Global i = i_start + lr
                    # Global k = k_start + lc
                    gi = i_start + lr
                    gk = k_start + lc
                    coeff_w = sign * coeff_res
                    if coeff_w != 0:
                        # W index is k*n + i
                        w_list.append((gk, gi, coeff_w))
            
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # --- Part 2: 4x4 * 4x1 (5th column of B) ---
    # C[:, 4] = A * B[:, 4]
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    # This is 16 scalar mults.
    
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
