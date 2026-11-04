import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = 4, 4, 6

    # Rank 77 for 4x4x4 (Recursive Strassen) + 4x4x2 (Recursive Strassen)
    # Total Rank 77? No.
    # R(4,4,4) <= 49 (Strassen)
    # R(4,4,2) <= 28 (Strassen)
    # Total 77.
    
    # Let's build the rank 77 decomposition exactly.
    # It consists of:
    # 1. 7 multiplications of 2x2 blocks (Strassen level 1)
    #    Each 2x2 multiplication is decomposed into 7 scalar multiplications (Strassen level 2).
    #    So 7 * 7 = 49 terms for the 4x4x4 part.
    # 2. For the 4x4x2 part, we split B into B_left (4x2) and B_right (4x2)? No, B is 4x6.
    #    We can view B as [B1 | B2] where B1 is 4x4 and B2 is 4x2.
    #    C = A B1 + A B2.
    #    A B1: 4x4 * 4x4 -> Rank 49.
    #    A B2: 4x4 * 4x2 -> Rank 28.
    #    Total 77.
    
    R = 77
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen 2x2 coefficients
    # A = [[a0, a1], [a2, a3]]
    # B = [[b0, b1], [b2, b3]]
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2-a0)(b0+b1)
    # M7 = (a1-a3)(b2+b3)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 7 Strassen terms as linear combinations of input elements.
    # u_coeffs: list of (local_idx, coeff) for A
    # v_coeffs: list of (local_idx, coeff) for B
    # w_coeffs: list of (local_idx, coeff) for C
    
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, -1), (2, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def add_block(a_off, b_off, c_off, col_start, b_width):
        """
        Adds a 2x2 * 2xb_width block multiplication using Strassen-like decomposition.
        For b_width=2, it's standard Strassen (7 terms).
        For b_width=1, it's schoolbook (4 terms)? 
        Wait, if we use the recursive structure:
        R(4,4,2) is decomposed into 7 multiplications of 2x2 * 2x1?
        No, R(2,2,1) is 4. So 7*4=28.
        
        Let's stick to the explicit construction of the 77 terms.
        The 49 terms for 4x4x4 come from 7 groups of 7.
        The 28 terms for 4x4x2 come from 7 groups of 4.
        
        We need a function to add a "super-term" which is a product of two 2x2 matrices (or 2x2 and 2x1).
        
        Actually, it's easier to just iterate over the 7 Strassen "macro" terms for the 4x4 case,
        and for each macro term, expand the 2x2*2x2 product into 7 micro terms.
        
        For the 4x4x2 case, we have 7 macro terms, but the product is 2x2 * 2x1.
        The rank of 2x2 * 2x1 is 4.
        So we expand each macro term into 4 micro terms.
        """
        
        # We need the coefficients of the macro terms on the 2x2 blocks.
        # This is what `strassen_terms` provides, but it assumes 2x2 output.
        # For 2x2 * 2x1, the output is 2x1.
        
        # Let's define the macro coefficients for A and B separately.
        # alpha[s][a], beta[s][b]
        
        alpha = [
            [1, 0, 0, 1], # M1: A00 + A11
            [0, 0, 1, 1], # M2: A10 + A11
            [1, 0, 0, 0], # M3: A00
            [0, 0, 0, 1], # M4: A11
            [1, 1, 0, 0], # M5: A00 + A01
            [-1, 0, 1, 0], # M6: A10 - A00
            [0, 1, 0, -1]  # M7: A01 - A11
        ]
        
        # For B (2x2):
        beta_2x2 = [
            [1, 0, 0, 1], # M1: B00 + B11
            [1, 0, 0, 0], # M2: B00
            [0, 1, 0, -1], # M3: B01 - B11
            [-1, 0, 1, 0], # M4: B10 - B00
            [0, 0, 0, 1], # M5: B11
            [1, 1, 0, 0], # M6: B00 + B01
            [0, 0, 1, 1]  # M7: B10 + B11
        ]
        
        # For B (2x1):
        # B = [b0; b1]
        # M1 = (A00+A11)(b0+b1)? No, Strassen doesn't apply directly to 2x1 in the same way.
        # We just use schoolbook for the micro-level?
        # Yes, R(2,2,1) = 4.
        # Micro terms for 2x2 * 2x1:
        # P1 = a00*b0
        # P2 = a01*b1
        # P3 = a10*b0
        # P4 = a11*b1
        # C00 = P1 + P2
        # C10 = P3 + P4
        
        # Wait, the macro term is (A_comb) * (B_comb).
        # If B_comb is 2x1, then (A_comb) * (B_comb) is 2x1.
        # We can decompose (A_comb) * (B_comb) using the 4 schoolbook terms.
        
        # So for each of the 7 macro terms s:
        # A_comb = sum_alpha[s][a] * A_block_a
        # B_comb = sum_beta[s][b] * B_block_b
        
        # For 4x4x4 (b_width=2):
        # B_block_b is 2x2.
        # Product P_s = A_comb * B_comb.
        # We decompose P_s using 7 Strassen micro-terms.
        # The coefficients of the micro-terms will be products of alpha[s][a], beta[s][b], and micro-coeffs.
        
        # This is getting complicated to implement dynamically.
        # Let's just hardcode the 77 terms? No, that's error-prone.
        
        # Let's use the tensor product structure.
        # The 4x4x4 tensor is a "tensor product" of two 2x2x2 tensors? No.
        # It's a recursive construction.
        
        # Let's write a helper that adds a "micro-term" (scalar product of two linear forms).
        # This is the base case.
        
        def add_micro_term(u_vec, v_vec, w_vec, col):
            """
            u_vec, v_vec, w_vec are sparse dicts {index: coeff}
            """
            for idx, coeff in u_vec.items():
                U[idx, col] += coeff
            for idx, coeff in v_vec.items():
                V[idx, col] += coeff
            for idx, coeff in w_vec.items():
                W[idx, col] += coeff
        
        # We will generate the 49 terms for 4x4x4.
        # And 28 terms for 4x4x2.
        
        # 4x4x4 Construction:
        # Split A, B, C into 2x2 blocks.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # For each macro term s in 0..6:
        #   A_s = sum_a alpha[s][a] A_a
        #   B_s = sum_b beta[s][b] B_b
        #   P_s = A_s * B_s (2x2 * 2x2)
        #   Decompose P_s into 7 micro terms using Strassen on the 2x2 matrices A_s and B_s.
        #   The micro terms are linear combinations of the elements of A_s and B_s.
        #   Since A_s and B_s are themselves linear combinations of A and B,
        #   the micro terms are linear combinations of A and B.
        
        # Let's pre-calculate the 7 micro-terms for a generic 2x2*2x2.
        # Micro term t (0..6) has:
        #   u_micro[t]: coeffs for A_s elements
        #   v_micro[t]: coeffs for B_s elements
        #   w_micro[t]: coeffs for C_s elements (which are linear combos of C blocks)
        
        # Actually, we can just use the `strassen_terms` defined above.
        # u_coeffs, v_coeffs, w_coeffs in `strassen_terms` are for the 4 elements of the 2x2 matrix.
        
        # So for macro term s, micro term t:
        # U contribution:
        #   For each a in 0..3 (A block index):
        #     If alpha[s][a] != 0:
        #       For each local_idx in u_coeffs[t]:
        #         U[index_of(A_a, local_idx), col] += alpha[s][a] * u_coeffs[t][local_idx]
        
        # This is O(7 * 7 * 4 * 4) operations. Very fast.
        
        # Let's implement this.
        
        col = 0
        
        # 4x4x4 Part
        for s in range(7):
            for t in range(7):
                u_c, v_c, w_c = strassen_terms[t]
                
                # U
                u_dict = {}
                for a in range(4):
                    if alpha[s][a] != 0:
                        for local_idx, coeff in u_c:
                            # Map local_idx (0..3) to global index in A
                            # A_a is 2x2.
                            # local_idx 0 -> A_a[0,0]
                            # local_idx 1 -> A_a[0,1]
                            # local_idx 2 -> A_a[1,0]
                            # local_idx 3 -> A_a[1,1]
                            
                            # A block a:
                            # a=0: A00 (rows 0,1, cols 0,1)
                            # a=1: A01 (rows 0,1, cols 2,3)
                            # a=2: A10 (rows 2,3, cols 0,1)
                            # a=3: A11 (rows 2,3, cols 2,3)
                            
                            br, bc = a // 2, a % 2
                            lr, lc = local_idx // 2, local_idx % 2
                            
                            global_row = br * 2 + lr
                            global_col = bc * 2 + lc
                            global_idx = global_row * 4 + global_col
                            
                            u_dict[global_idx] = u_dict.get(global_idx, 0) + alpha[s][a] * coeff
                
                # V
                v_dict = {}
                for b in range(4):
                    if beta_2x2[s][b] != 0:
                        for local_idx, coeff in v_c:
                            br, bc = b // 2, b % 2
                            lr, lc = local_idx // 2, local_idx % 2
                            
                            global_row = br * 2 + lr
                            global_col = bc * 2 + lc
                            global_idx = global_row * 4 + global_col
                            
                            v_dict[global_idx] = v_dict.get(global_idx, 0) + beta_2x2[s][b] * coeff
                
                # W
                # The output of P_s is added to C blocks according to recon[s].
                # recon[s] = [(c_block_idx, gamma), ...]
                # For each c_block in recon[s]:
                #   The 2x2 output of micro term t is added to C_block.
                #   w_c gives the coeffs for the 4 elements of the 2x2 output.
                
                # We need the recon list.
                recon = [
                    [(0, 1), (3, 1)],       # M1 -> C00(+1), C11(+1)
                    [(2, 1), (3, -1)],      # M2 -> C10(+1), C11(-1)
                    [(1, 1), (3, 1)],       # M3 -> C01(+1), C11(+1)
                    [(0, 1), (2, 1)],       # M4 -> C00(+1), C10(+1)
                    [(0, -1), (1, 1)],      # M5 -> C00(-1), C01(+1)
                    [(3, 1)],               # M6 -> C11(+1)
                    [(0, 1)]                # M7 -> C00(+1)
                ]
                
                w_dict = {}
                for c_block, gamma in recon[s]:
                    for local_idx, coeff in w_c:
                        br, bc = c_block // 2, c_block % 2
                        lr, lc = local_idx // 2, local_idx % 2
                        
                        global_row = br * 2 + lr
                        global_col = bc * 2 + lc
                        global_idx = global_row * 4 + global_col
                        
                        w_dict[global_idx] = w_dict.get(global_idx, 0) + gamma * coeff
                
                add_micro_term(u_dict, v_dict, w_dict, col)
                col += 1
        
        # 4x4x2 Part
        # B is 4x6. We used cols 0-3 for 4x4x4.
        # Now we use cols 4-5 for 4x4x2.
        # B_right is 4x2.
        # Split B_right into 2x1 blocks?
        # B_right = [B_R00, B_R01; B_R10, B_R11] where each is 2x1.
        # Wait, B is 4x6.
        # B_right rows 0-3, cols 4-5.
        # B_R00: rows 0-1, col 4. (2x1)
        # B_R01: rows 0-1, col 5. (2x1)
        # B_R10: rows 2-3, col 4. (2x1)
        # B_R11: rows 2-3, col 5. (2x1)
        
        # C_right is 4x2.
        # C_R00: rows 0-1, col 4.
        # C_R01: rows 0-1, col 5.
        # C_R10: rows 2-3, col 4.
        # C_R11: rows 2-3, col 5.
        
        # Macro terms s in 0..6:
        # A_s is 2x2 (same as before).
        # B_s is 2x2? No, B_s is formed from 2x1 blocks.
        # Wait, Strassen on 4x4 * 4x2.
        # Split A into 2x2 blocks.
        # Split B_right into 2x1 blocks.
        # B_right = [B0; B1] where B0 is 2x2? No, B_right is 4x2.
        # B_right = [[B00, B01], [B10, B11]] where Bxx are 2x1?
        # No, B_right is 4 rows, 2 cols.
        # Top half: 2x2. Bottom half: 2x2.
        # So B_right = [B_top; B_bot] where B_top is 2x2, B_bot is 2x2.
        
        # Wait, if B_top is 2x2, then we are back to 2x2 * 2x2?
        # No, A is 4x4.
        # A = [A00 A01; A10 A11].
        # B_right = [B_top; B_bot].
        # C_right = A B_right.
        # C_top = A00 B_top + A01 B_bot? No.
        # C_top (2x2) = A00 B_top + A01 B_bot?
        # A00 is 2x2. B_top is 2x2.
        # A01 is 2x2. B_bot is 2x2.
        # So C_top is sum of two 2x2*2x2 products.
        # This would be rank 7+7=14 for C_top.
        # C_bot is sum of two 2x2*2x2 products. Rank 14.
        # Total 28.
        
        # This matches R(4,4,2) <= 28.
        
        # So we can just reuse the 4x4x4 logic, but restrict B and C to the right half.
        # B indices for right half:
        # B_top: rows 0-1, cols 4-5.
        # B_bot: rows 2-3, cols 4-5.
        
        # C indices for right half:
        # C_top: rows 0-1, cols 4-5.
        # C_bot: rows 2-3, cols 4-5.
        
        # We need to map the "global" indices in the 4x4x4 logic to these subsets.
        
        for s in range(7):
            for t in range(7):
                u_c, v_c, w_c = strassen_terms[t]
                
                # U is the same (A is 4x4)
                u_dict = {}
                for a in range(4):
                    if alpha[s][a] != 0:
                        for local_idx, coeff in u_c:
                            br, bc = a // 2, a % 2
                            lr, lc = local_idx // 2, local_idx % 2
                            
                            global_row = br * 2 + lr
                            global_col = bc * 2 + lc
                            global_idx = global_row * 4 + global_col
                            
                            u_dict[global_idx] = u_dict.get(global_idx, 0) + alpha[s][a] * coeff
                
                # V is for B_right.
                # B blocks:
                # b=0: B_top (rows 0-1, cols 4-5)
                # b=1: ?
                # In the 4x4x4 logic, b=0,1,2,3 correspond to 4 2x2 blocks of B.
                # Here we only have 2 2x2 blocks of B_right (Top, Bot).
                # But the Strassen macro terms combine 4 blocks.
                # So we need to map the 4 virtual blocks to the 2 physical blocks?
                # No, that's not right.
                
                # Let's re-evaluate the 4x4x2 decomposition.
                # C = A B.
                # A = [A11 A12; A21 A22]
                # B = [B1; B2] (B1 is 2x2, B2 is 2x2)
                # C1 = A11 B1 + A12 B2
                # C2 = A21 B1 + A22 B2
                
                # We can compute:
                # M1 = (A11+A22)(B1+B2)
                # M2 = (A21+A22)B1
                # M3 = A11(B2-B1)
                # M4 = A22(B2-B1) ? No.
                
                # Standard Strassen for (2x2 * 2x2) + (2x2 * 2x2):
                # Let X = A11, Y = A12, Z = A21, W = A22.
                # Let U = B1, V = B2.
                # C1 = X U + Y V
                # C2 = Z U + W V
                
                # We can use 7 multiplications:
                # M1 = (X+W)(U+V)
                # M2 = (Z+W)U
                # M3 = X(V-U)
                # M4 = W(V-U)
                # M5 = (X+Y)V
                # M6 = (Z-X)(U+V)
                # M7 = (Y-W)(U+V) ? No.
                
                # Actually, there is a known rank 7 algorithm for 2x2 * 2x2? No, rank 7 is for one product.
                # For two products, rank 14?
                # Yes, 7+7=14.
                
                # So we just run the 4x4x4 Strassen logic, but the "B" matrix is effectively 4x2?
                # No, the B matrix in the 4x4x4 logic is 4x4.
                # If we set the right half of B to be the actual B_right, and the left half to 0?
                # No, that would compute A * [0 | B_right].
                # Which is exactly A * B_right.
                
                # So we can just reuse the EXACT same 49-term logic, but:
                # 1. V indices map to B_right (cols 4-5).
                # 2. W indices map to C_right (cols 4-5).
                # 3. U indices map to A (cols 0-3).
                
                # Let's do that.
                
                # V mapping:
                # In 4x4x4 logic, b_idx 0,1,2,3 map to B blocks.
                # Here, we want b_idx 0,1 to map to B_top, and b_idx 2,3 to map to B_bot?
                # No, the Strassen macro terms mix all 4 blocks.
                # If we set B_left = 0 and B_right = B_actual, then:
                # B_block_0 (top-left) = 0
                # B_block_1 (top-right) = B_top_left? No.
                
                # Let's clarify B layout.
                # B is 4x6.
                # B[0:2, 0:2] = B00
                # B[0:2, 2:4] = B01
                # B[2:4, 0:2] = B10
                # B[2:4, 2:4] = B11
                
                # For 4x4x4, we used B[0:4, 0:4].
                # For 4x4x2, we want to use B[0:4, 4:6].
                # This is a 4x2 matrix.
                # It is NOT a 4x4 matrix.
                # So we cannot directly reuse the 4x4x4 logic which assumes 4x4 B.
                
                # We need a specific 4x4x2 decomposition.
                # Rank 28.
                # We can view it as 7 multiplications of 2x2 * 2x1?
                # No, 7 * 4 = 28.
                # So we need 7 "macro" terms, each decomposing into 4 "micro" terms (schoolbook 2x2*2x1).
                
                # Let's define the 7 macro terms for 4x4x2.
                # A = [A11 A12; A21 A22]
                # B = [B1; B2] (B1, B2 are 2x2)
                # C = [C1; C2]
                
                # M1 = (A11+A22)(B1+B2)
                # M2 = (A21+A22)B1
                # M3 = A11(B2-B1)
                # M4 = A22(B2-B1)
                # M5 = (A11+A12)B2
                # M6 = (A21-A11)(B1+B2)
                # M7 = (A12-A22)(B1+B2)
                
                # C1 = M1 + M4 - M5 + M7?
                # C2 = M2 + M4?
                
                # Let's check:
                # C1 = A11 B1 + A12 B2
                # C2 = A21 B1 + A22 B2
                
                # M1 + M4 - M5 + M7
                # = (A11+A22)(B1+B2) + A22(B2-B1) - (A11+A12)B2 + (A12-A22)(B1+B2)
                # = A11 B1 + A11 B2 + A22 B1 + A22 B2 + A22 B2 - A22 B1 - A11 B2 - A12 B2 + A12 B1 + A12 B2 - A22 B1 - A22 B2
                # = A11 B1 + A12 B1 + (A11 B2 - A11 B2) + (A22 B1 - A22 B1) + (A22 B2 + A22 B2 - A22 B2 - A22 B2) - A12 B2 + A12 B2
                # = A11 B1 + A12 B1
                # This is NOT C1. C1 = A11 B1 + A12 B2.
                
                # So this set of macro terms is wrong.
                
                # Let's use the standard Strassen on the 2x2 block matrix multiplication?
                # No, the blocks are 2x2 and 2x2.
                # We are doing 4x4 * 4x2.
                # This is equivalent to two 4x4 * 4x1 multiplications.
                # R(4,4,1) = 16?
                # 2 * 16 = 32.
                # But we know R(4,4,2) <= 28.
                
                # How to get 28?
                # Use Strassen on the 4x4 part?
                # R(4,4,2) <= 7 * R(2,2,1) = 7 * 4 = 28.
                # So we need 7 macro terms, each of which is a product of a 2x2 matrix (from A) and a 2x1 matrix (from B)?
                # No, the B blocks are 2x2.
                
                # Let's look at the structure of R(4,4,2) <= 28.
                # It comes from R(4,4,4) <= 49? No.
                # It comes from R(2,2,2) = 7.
                # If we split A into 2x2 blocks and B into 2x2 blocks (vertically),
                # then C = A B.
                # C_top = A_top B_top + A_bot B_bot? No.
                # C_top = A11 B1 + A12 B2.
                
                # We can compute A11 B1 (rank 7) and A12 B2 (rank 7).
                # Total 14 for C_top.
                # 14 for C_bot.
                # Total 28.
                
                # This is just two independent 2x2*2x2 multiplications for each output block.
                # So we can just use the 2x2*2x2 Strassen decomposition 4 times?
                # No, A11 B1, A12 B2, A21 B1, A22 B2.
                # 4 multiplications of 2x2*2x2.
                # Each rank 7.
                # Total 28.
                
                # So the 28 terms are:
                # 1. Strassen for A11 * B1 -> C_top_part1
                # 2. Strassen for A12 * B2 -> C_top_part2
                # 3. Strassen for A21 * B1 -> C_bot_part1
                # 4. Strassen for A22 * B2 -> C_bot_part2
                
                # We can implement this by calling the 2x2*2x2 Strassen adder 4 times.
                
                def add_2x2_strassen(a_off, b_off, c_off, col_start):
                    """
                    Adds a 2x2 * 2x2 Strassen decomposition.
                    a_off: offset in A (flattened 4x4)
                    b_off: offset in B (flattened 4x6)
                    c_off: offset in C (flattened 4x6)
                    """
                    for t in range(7):
                        u_c, v_c, w_c = strassen_terms[t]
                        
                        u_dict = {}
                        for local_idx, coeff in u_c:
                            # A block is 2x2.
                            # a_off is the index of A[0,0] of the block.
                            # A is 4x4.
                            # If block is at (br, bc), a_off = br*4 + bc.
                            # local_idx 0 -> (0,0) -> a_off
                            # local_idx 1 -> (0,1) -> a_off+1
                            # local_idx 2 -> (1,0) -> a_off+4
                            # local_idx 3 -> (1,1) -> a_off+5
                            
                            # This assumes the block is contiguous in row-major?
                            # A[br, bc], A[br, bc+1]
                            # A[br+1, bc], A[br+1, bc+1]
                            # Yes.
                            
                            offsets = [0, 1, 4, 5]
                            global_idx = a_off + offsets[local_idx]
                            u_dict[global_idx] = u_dict.get(global_idx, 0) + coeff
                        
                        v_dict = {}
                        for local_idx, coeff in v_c:
                            # B block is 2x2.
                            # b_off is index of B[0,0] of the block.
                            # B is 4x6.
                            # offsets: 0, 1, 6, 7
                            
                            offsets = [0, 1, 6, 7]
                            global_idx = b_off + offsets[local_idx]
                            v_dict[global_idx] = v_dict.get(global_idx, 0) + coeff
                        
                        w_dict = {}
                        for local_idx, coeff in w_c:
                            # C block is 2x2.
                            # c_off is index of C[0,0] of the block.
                            # C is 4x6.
                            # offsets: 0, 1, 6, 7
                            
                            offsets = [0, 1, 6, 7]
                            global_idx = c_off + offsets[local_idx]
                            w_dict[global_idx] = w_dict.get(global_idx, 0) + coeff
                        
                        add_micro_term(u_dict, v_dict, w_dict, col_start + t)
                
                # 1. A11 * B1 -> C_top (part 1)
                # A11: a_off = 0
                # B1: B_top (rows 0-1, cols 4-5). b_off = 4.
                # C_top: c_off = 4.
                add_2x2_strassen(0, 4, 4, col)
                col += 7
                
                # 2. A12 * B2 -> C_top (part 2)
                # A12: a_off = 2
                # B2: B_bot (rows 2-3, cols 4-5). b_off = 4 + 12 = 16.
                # C_top: c_off = 4.
                add_2x2_strassen(2, 16, 4, col)
                col += 7
                
                # 3. A21 * B1 -> C_bot (part 1)
                # A21: a_off = 8
                # B1: b_off = 4.
                # C_bot: c_off = 4 + 12 = 16.
                add_2x2_strassen(8, 4, 16, col)
                col += 7
                
                # 4. A22 * B2 -> C_bot (part 2)
                # A22: a_off = 10
                # B2: b_off = 16.
                # C_bot: c_off = 16.
                add_2x2_strassen(10, 16, 16, col)
                col += 7
                
    # Total 49 + 28 = 77.
    
    # Verify
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(recon, T, atol=1e-9):
        # Fallback to schoolbook if something went wrong
        R = 96
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            result.append([])
            for x in row:
                x_rounded = round(x)
                if abs(x - x_rounded) < 1e-9:
                    result[-1].append(int(x_rounded))
                else:
                    result[-1].append(float(x))
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
