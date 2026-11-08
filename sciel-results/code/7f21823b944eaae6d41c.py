import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    deadline = start_time + 110.0

    # Total rank estimate: 28 (for 4x4x4) + 48 (for 4x4x3) = 76
    total_rank = 76
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0

    # --- Part 1: 4x4x4 (k in 0..3) ---
    # Strassen decomposition of 4x4x4 using 2x2 blocks.
    # A blocks: A00 (0:2,0:2), A01 (0:2,2:4), A10 (2:4,0:2), A11 (2:4,2:4)
    # B blocks: B00 (0:2,0:2), B01 (0:2,2:4), B10 (2:4,0:2), B11 (2:4,2:4)
    
    # Strassen terms definitions:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # We need to add linear combinations to U, V, W.
    # For a term "Sum_A * Sum_B", we expand Sum_A into 4 scalar terms and Sum_B into 4 scalar terms.
    # Each scalar term is a rank-1 tensor for the 2x2x2 sub-problem.
    
    # Helper to add a 2x2x2 schoolbook term to U, V, W at given col
    # a_idx: (row, col) in A block
    # b_idx: (row, col) in B block
    # c_idx: (row, col) in C block (k in 0..3 for this part)
    def add_2x2_term(col_idx, a_r, a_c, b_r, b_c, c_r, c_k, coeff=1.0):
        # A index: global i = block_row*2 + a_r, global j = block_col*2 + a_c
        # Wait, A is 4x4. Blocks are 2x2.
        # A[bi*2+ai, bj*2+aj]
        # B is 4x7. For this part, k is 0..3.
        # B[j, k]. j = bj*2 + b_j_local? No.
        # B block B_{ij} is A_block_i times B_block_j.
        # B block B_{ij} has rows i*2..i*2+1 and cols j*2..j*2+1?
        # No, B is 4x7. The 4x4 part is B[:, 0:4].
        # So B_block_{ij} is B[i*2:(i+1)*2, j*2:(j+1)*2].
        # So global j (row of B) = i*2 + b_r_local.
        # Global k (col of B) = j*2 + b_c_local.
        #
        # A block A_{ij} is A[i*2:(i+1)*2, j*2:(j+1)*2].
        # Global i (row of A) = i*2 + a_r_local.
        # Global j (col of A) = j*2 + a_c_local.
        #
        # C block C_{ij} is C[i*2:(i+1)*2, j*2:(j+1)*2].
        # Global i (row of C) = i*2 + c_r_local.
        # Global k (col of C) = j*2 + c_k_local.
        
        # Inputs to this helper:
        # a_r, a_c: local indices in A block (0..1)
        # b_r, b_c: local indices in B block (0..1)
        # c_r, c_k: local indices in C block (0..1)
        # block_i, block_j, block_k: block indices? 
        # The helper needs to know which block we are in to map to global indices.
        # Actually, the expansion of M_l = X_l Y_l where X_l is a linear comb of A blocks and Y_l of B blocks.
        # It's easier to iterate through the 7 Strassen terms and their constituent blocks.
        pass

    # Let's define the Strassen terms more explicitly with block indices.
    # Term 1: (A00+A11) * (B00+B11)
    # Term 2: (A10+A11) * B00
    # Term 3: A00 * (B10-B11)
    # Term 4: A11 * (B01-B00)
    # Term 5: (A00+A10) * B11
    # Term 6: (A10-A00) * (B00+B10)
    # Term 7: (A01-A11) * (B01+B11)
    
    # For each term, we have a sum of A blocks and a sum of B blocks.
    # We iterate through each pair (A_block, B_block) with its coefficient.
    # For each pair, we do schoolbook 2x2x2 (4 terms).
    
    strassen_terms = [
        # (coeff_A, block_i, block_j), (coeff_B, block_i, block_j)
        # Term 1
        ([(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)]),
        # Term 2
        ([(1, 1, 0), (1, 1, 1)], [(1, 0, 0)]),
        # Term 3
        ([(1, 0, 0)], [(1, 1, 0), (-1, 1, 1)]),
        # Term 4
        ([(1, 1, 1)], [(1, 0, 1), (-1, 0, 0)]),
        # Term 5
        ([(1, 0, 0), (1, 1, 0)], [(1, 1, 1)]),
        # Term 6
        ([(1, 1, 0), (-1, 0, 0)], [(1, 0, 0), (1, 1, 0)]),
        # Term 7
        ([(1, 0, 1), (-1, 1, 1)], [(1, 0, 1), (1, 1, 1)])
    ]
    
    # Function to add a single 2x2x2 schoolbook decomposition
    # a_block_i, a_block_j: A block indices
    # b_block_i, b_block_j: B block indices (B block i,j corresponds to rows i*2..i*2+1, cols j*2..j*2+1 of B_1)
    # coeff: coefficient
    def add_schoolbook_2x2x2(col_idx, a_bi, a_bj, b_bi, b_bj, coeff):
        # Iterate over local indices
        for lr in range(2):
            for lc in range(2):
                # A local index: lr, lc
                # Global A index: i = a_bi*2 + lr, j = a_bj*2 + lc
                # U index: i*m + j
                u_idx = (a_bi*2 + lr)*m + (a_bj*2 + lc)
                U[u_idx, col_idx] += coeff * 1.0 # A part is just 1
                
                # B local index: lr, lc (same local structure)
                # Global B index for B_1: row = b_bi*2 + lr, col = b_bj*2 + lc
                # V index: row*p + col
                v_idx = (b_bi*2 + lr)*p + (b_bj*2 + lc)
                V[v_idx, col_idx] += coeff * 1.0 # B part is just 1
                
                # C local index: lr, lc
                # Global C index for B_1 part: i = a_bi*2 + lr, k = b_bj*2 + lc
                # W index: k*n + i
                w_idx = (b_bj*2 + lc)*n + (a_bi*2 + lr)
                W[w_idx, col_idx] += coeff * 1.0 # C part is just 1
                col_idx += 1
        return col_idx

    # Process Strassen terms
    for a_terms, b_terms in strassen_terms:
        for c_a, a_bi, a_bj in a_terms:
            for c_b, b_bi, b_bj in b_terms:
                coeff = c_a * c_b
                col = add_schoolbook_2x2x2(col, a_bi, a_bj, b_bi, b_bj, coeff)

    # --- Part 2: 4x4x3 (k in 4..6) ---
    # A is 4x4. B_2 is 4x3.
    # Split B_2 into B_20 (rows 0:2, cols 4:7) and B_21 (rows 2:4, cols 4:7).
    # C_20 = A_00 B_20 + A_01 B_21 (rows 0:2 of C)
    # C_21 = A_10 B_20 + A_11 B_21 (rows 2:4 of C)
    
    # We use schoolbook for each 2x2 (A block) * 2x3 (B block) multiplication.
    # Rank of 2x2x3 is 12.
    # 4 such multiplications -> 48 terms.
    
    def add_schoolbook_2x2x3(col_idx, a_bi, a_bj, b_row_offset, coeff):
        # a_bi, a_bj: A block indices (2x2)
        # b_row_offset: 0 for B_20, 1 for B_21 (determines row of B)
        # B_20 is rows 0:2, cols 4:7.
        # B_21 is rows 2:4, cols 4:7.
        # C_20 is rows 0:2, cols 4:7.
        # C_21 is rows 2:4, cols 4:7.
        
        # A local indices: lr (0:1), lc (0:1)
        # Global A: i = a_bi*2 + lr, j = a_bj*2 + lc
        # U index: i*m + j
        
        # B local indices: lr_b (0:1), lc_b (0:2) for cols
        # Global B: row = b_row_offset*2 + lr_b, col = 4 + lc_b
        # V index: row*p + col
        
        # C local indices: lr_c (0:1), lc_c (0:2) for cols
        # Global C: i = a_bi*2 + lr_c, k = 4 + lc_c
        # W index: k*n + i
        
        for lr_a in range(2):
            for lc_a in range(2):
                u_idx = (a_bi*2 + lr_a)*m + (a_bj*2 + lc_a)
                
                for lr_b in range(2):
                    for lc_b in range(3):
                        v_idx = (b_row_offset*2 + lr_b)*p + (4 + lc_b)
                        
                        # C index corresponds to A row and B col
                        w_idx = (4 + lc_b)*n + (a_bi*2 + lr_a)
                        
                        U[u_idx, col_idx] += coeff
                        V[v_idx, col_idx] += coeff
                        W[w_idx, col_idx] += coeff
                        col_idx += 1
        return col_idx

    # Term 1: A_00 * B_20
    col = add_schoolbook_2x2x3(col, 0, 0, 0, 1.0)
    # Term 2: A_01 * B_21
    col = add_schoolbook_2x2x3(col, 0, 1, 1, 1.0)
    # Term 3: A_10 * B_20
    col = add_schoolbook_2x2x3(col, 1, 0, 0, 1.0)
    # Term 4: A_11 * B_21
    col = add_schoolbook_2x2x3(col, 1, 1, 1, 1.0)
    
    # Verify column count
    # Part 1: 7 terms.
    # T1: 2 A * 2 B = 4 blocks * 4 terms = 16
    # T2: 2 A * 1 B = 2 blocks * 4 terms = 8
    # T3: 1 A * 2 B = 2 blocks * 4 terms = 8
    # T4: 1 A * 2 B = 2 blocks * 4 terms = 8
    # T5: 2 A * 1 B = 2 blocks * 4 terms = 8
    # T6: 2 A * 2 B = 4 blocks * 4 terms = 16
    # T7: 2 A * 2 B = 4 blocks * 4 terms = 16
    # Total Part 1: 16+8+8+8+8+16+16 = 80?
    # Wait, my previous estimate was 28. Why?
    # Strassen 7 terms. Each term is a 2x2x2 multiplication.
    # 2x2x2 schoolbook is 4 terms.
    # 7 * 4 = 28.
    # Where did I go wrong?
    # Ah, the Strassen terms are linear combinations.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This expands to 4 products.
    # Each product is a 2x2x2 multiplication (4 terms).
    # So M1 contributes 16 terms.
    # M2 = (A10+A11)B00 = A10B00 + A11B00. 2 products -> 8 terms.
    # M3 = A00(B10-B11) = A00B10 - A00B11. 2 products -> 8 terms.
    # M4 = A11(B01-B00) = A11B01 - A11B00. 2 products -> 8 terms.
    # M5 = (A00+A10)B11 = A00B11 + A10B11. 2 products -> 8 terms.
    # M6 = (A10-A00)(B00+B10) = A10B00 + A10B10 - A00B00 - A00B10. 4 products -> 16 terms.
    # M7 = (A01-A11)(B01+B11) = A01B01 + A01B11 - A11B01 - A11B11. 4 products -> 16 terms.
    # Total: 16+8+8+8+8+16+16 = 80.
    
    # So the rank of this specific Strassen-based decomposition is 80.
    # This is still better than 112.
    # Part 2 is 48.
    # Total rank = 80 + 48 = 128.
    # This is WORSE than 112.
    
    # I made a mistake in the estimation. The "rank 28" comes from the number of scalar multiplications in Strassen, but each scalar multiplication here is a 2x2x2 tensor which has rank 4.
    # So rank is 7 * 4 = 28? NO.
    # Strassen reduces the number of *bilinear forms* to 7.
    # But each bilinear form is a 2x2x2 multiplication.
    # If we decompose each 2x2x2 into rank 4, we get 7 * 4 = 28 rank-1 terms.
    # BUT, the Strassen formula involves sums like (A00+A11).
    # (A00+A11)B00 = A00B00 + A11B00.
    # This is two separate 2x2x2 multiplications.
    # So the rank is the sum of the ranks of the constituent 2x2x2 multiplications.
    # So my calculation of 80 is correct for this implementation.
    
    # Is there a better way?
    # Yes, use the rank-13 decomposition for 4x4x4 directly if possible, or a better Strassen variant.
    # But 128 is worse than 112. I should fall back to schoolbook 112 if I can't find better.
    # Or, I can just use schoolbook for the 4x4x4 part (rank 64) and schoolbook for 4x4x3 (rank 48).
    # Total 112.
    
    # Let's try to optimize the 4x4x3 part.
    # If I use Strassen for the 2x2x2 parts of the 2x2x3?
    # No, 2x2x3 doesn't have a simple Strassen.
    
    # Let's check if I can use the rank-13 decomposition for 4x4x4.
    # Since I don't have it, I will stick to schoolbook to ensure validity and not regress below 112.
    # Actually, I can just return the schoolbook 112.
    
    # Wait, is 80+48=128 really the best I can do with this structure?
    # What if I just use schoolbook for the whole thing? 112.
    # 112 < 128.
    # So I will implement the schoolbook decomposition.
    
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    rank = n * m * p
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}

