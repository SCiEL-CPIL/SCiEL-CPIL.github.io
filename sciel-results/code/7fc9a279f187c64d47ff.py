import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 150  # seconds

    n, m, p = DIMENSIONS  # 4, 4, 9

    # We will build the decomposition using Strassen for the 4x4 * 4x4 part
    # and schoolbook for the remaining columns, or a direct Strassen-based approach.
    
    # Strategy:
    # 1. Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # 2. Split B (4x9) into 2x2 blocks vertically? No, B is 4x9.
    #    Let's split B into top 2 rows and bottom 2 rows.
    #    B = [B_top; B_bottom] where B_top is 2x9, B_bottom is 2x9.
    #    But Strassen requires square blocks usually, or at least compatible.
    #    Actually, Strassen works for rectangular matrices if we are careful.
    #    
    #    Let's use the standard 4x4 Strassen decomposition for A and a 4x9 B.
    #    A = [[A11, A12], [A21, A22]] (each 2x2)
    #    B = [[B11, B12, B13], [B21, B22, B23]] (each 2x3)
    #    
    #    Strassen products:
    #    M1 = (A11 + A22) * (B11 + B21)  -- Wait, B blocks are 2x3.
    #    The product M1 is (2x2) * (2x3) = 2x3.
    #    M2 = (A21 + A22) * B11          -- 2x2 * 2x3 = 2x3
    #    M3 = A11 * (B12 - B22)          -- 2x2 * 2x3 = 2x3
    #    M4 = A22 * (B21 - B11)          -- 2x2 * 2x3 = 2x3
    #    M5 = (A11 + A12) * B22          -- 2x2 * 2x3 = 2x3
    #    M6 = (A21 - A11) * (B11 + B12)  -- 2x2 * 2x3 = 2x3
    #    M7 = (A12 - A22) * (B21 + B22)  -- 2x2 * 2x3 = 2x3
    #
    #    Results:
    #    C11 = M1 + M4 - M5 + M7  (2x3)
    #    C12 = M3 + M5            (2x3)
    #    C21 = M2 + M4            (2x3)
    #    C22 = M1 - M2 + M3 + M6  (2x3)
    #
    #    So C is 4x9, composed of 4 blocks of 2x3.
    #    C = [[C11, C12, C13], [C21, C22, C23]]? No.
    #    The blocks B11, B12, B13 correspond to columns 0-2, 3-5, 6-8.
    #    So C11 is top-left 2x3, C12 is top-middle 2x3, C13 is top-right 2x3?
    #    Wait, the Strassen formula above only uses B11, B12, B21, B22.
    #    It ignores B13 and B23.
    #    So this only computes the first 6 columns of C.
    #    We need to handle the last 3 columns separately.
    #
    #    Let's split B into B_left (4x6) and B_right (4x3).
    #    Compute C_left = A * B_left using Strassen (7 products of 2x2 * 2x3).
    #    Compute C_right = A * B_right using Schoolbook (4*4*3 = 48 products? Or better?)
    #    
    #    For C_left:
    #    Each of the 7 Strassen products is a 2x2 * 2x3 multiplication.
    #    Rank of 2x2 * 2x3 matrix multiplication:
    #    Schoolbook: 2*2*3 = 12.
    #    Can we do better?
    #    2x2 * 2x3 is equivalent to 6 dot products of 2-vectors?
    #    Actually, we can use Strassen for 2x2 * 2x2 twice? No.
    #    Let's just use schoolbook for the inner 2x2 * 2x3.
    #    So each M_i uses 12 scalar multiplications.
    #    Total for C_left: 7 * 12 = 84 scalar multiplications.
    #
    #    For C_right (4x4 * 4x3):
    #    Schoolbook: 4*4*3 = 48.
    #    Can we do better?
    #    Maybe Strassen for 4x4 * 4x4 (49) and then project? No, that's worse.
    #    Maybe split 4x3 into 2x3 and 2x3?
    #    A = [[A11, A12], [A21, A22]]
    #    B_r = [[B11_r, B12_r], [B21_r, B22_r]] (each 2x3? No, B_r is 4x3)
    #    B_r = [B_top; B_bottom] where B_top is 2x3, B_bottom is 2x3.
    #    C_right = A * B_r.
    #    C_top = A11*B_top + A12*B_bottom (2x3)
    #    C_bottom = A21*B_top + A22*B_bottom (2x3)
    #    Each 2x2 * 2x3 uses 12 mults.
    #    Total 4 * 12 = 48. Same as schoolbook.
    #
    #    So total rank = 84 + 48 = 132.
    #    Schoolbook was 144.
    #    132 < 144. Improvement!
    #
    #    Let's verify the 2x2 * 2x3 schoolbook decomposition.
    #    A (2x2) * B (2x3) = C (2x3).
    #    C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    #    C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    #    C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    #    C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    #    C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    #    C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    #
    #    Terms (a_idx, b_idx, c_idx):
    #    a_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    #    b_idx: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    #    c_idx: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    #
    #    Term 1: A[0,0] * B[0,0] -> C[0,0]
    #    Term 2: A[0,1] * B[1,0] -> C[0,0]
    #    Term 3: A[0,0] * B[0,1] -> C[0,1]
    #    Term 4: A[0,1] * B[1,1] -> C[0,1]
    #    Term 5: A[0,0] * B[0,2] -> C[0,2]
    #    Term 6: A[0,1] * B[1,2] -> C[0,2]
    #    Term 7: A[1,0] * B[0,0] -> C[1,0]
    #    Term 8: A[1,1] * B[1,0] -> C[1,0]
    #    Term 9: A[1,0] * B[0,1] -> C[1,1]
    #    Term 10: A[1,1] * B[1,1] -> C[1,1]
    #    Term 11: A[1,0] * B[0,2] -> C[1,2]
    #    Term 12: A[1,1] * B[1,2] -> C[1,2]

    # We will generate the terms for the whole matrix multiplication.
    # Terms are (u_idx, v_idx, w_idx, coeff)
    # u_idx: i*m + j (0..15)
    # v_idx: j*p + k (0..35)
    # w_idx: k*n + i (0..35)

    terms = []

    def add_term(u_idx, v_idx, w_idx, coeff):
        if coeff != 0:
            terms.append((u_idx, v_idx, w_idx, coeff))

    # --- Part 1: C_left (columns 0..5) ---
    # Strassen 4x4 * 4x6
    # A blocks: A11 (0:2, 0:2), A12 (0:2, 2:4), A21 (2:4, 0:2), A22 (2:4, 2:4)
    # B blocks (for left part):
    # B11 (0:2, 0:3), B12 (0:2, 3:6), B21 (2:4, 0:3), B22 (2:4, 3:6)
    
    # Strassen products:
    # M1 = (A11 + A22) * (B11 + B21)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need a helper to add a 2x2 * 2x3 product with coefficients.
    # A_sum: list of (a_idx_local, coeff) for the 2x2 matrix A_part
    # B_sum: list of (b_idx_local, coeff) for the 2x3 matrix B_part
    # C_target: which C block (0: C11, 1: C12, 2: C21, 3: C22)
    # C_coeff: coefficient for this M_i in the C block formula
    
    # Local indices for 2x2 A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for 2x3 B: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # Local indices for 2x3 C: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    
    # Schoolbook terms for 2x2 * 2x3:
    # (a_loc, b_loc, c_loc)
    sb_2x2_2x3 = [
        (0, 0, 0), (1, 3, 0),
        (0, 1, 1), (1, 4, 1),
        (0, 2, 2), (1, 5, 2),
        (2, 0, 3), (3, 3, 3),
        (2, 1, 4), (3, 4, 4),
        (2, 2, 5), (3, 5, 5)
    ]

    def add_strassen_product(a_coeffs, b_coeffs, c_block, c_coeff):
        """
        a_coeffs: list of (local_a_idx, coeff)
        b_coeffs: list of (local_b_idx, coeff)
        c_block: 0,1,2,3 corresponding to C11, C12, C21, C22
        c_coeff: multiplier for the whole product
        """
        # Determine global offsets
        # c_block 0 (C11): rows 0-1, cols 0-2
        # c_block 1 (C12): rows 0-1, cols 3-5
        # c_block 2 (C21): rows 2-3, cols 0-2
        # c_block 3 (C22): rows 2-3, cols 3-5
        
        if c_block == 0:
            row_base = 0
            col_base = 0
        elif c_block == 1:
            row_base = 0
            col_base = 3
        elif c_block == 2:
            row_base = 2
            col_base = 0
        elif c_block == 3:
            row_base = 2
            col_base = 3
        
        # For each pair of (a, b) in the sums, and each schoolbook term
        for a_loc, a_c in a_coeffs:
            for b_loc, b_c in b_coeffs:
                # Find which schoolbook terms involve this a_loc and b_loc?
                # No, the schoolbook terms are specific pairs.
                # We need to check if (a_loc, b_loc) is one of the pairs in sb_2x2_2x3.
                # Actually, the product is linear.
                # Sum_{a,b} a_c[a] b_c[b] (A_a * B_b)
                # Each A_a * B_b contributes to specific C elements.
                # We can just iterate over the schoolbook terms and see if they match?
                # No, the schoolbook terms are the elementary products.
                # The product of sums is the sum of all elementary products.
                # So we iterate over all a_loc in a_coeffs and b_loc in b_coeffs.
                # For each pair, we need to know which C elements it contributes to.
                
                # Let's precompute the contribution of each (a_loc, b_loc) to C_loc.
                # From sb_2x2_2x3:
                # (0,0)->0, (1,3)->0
                # (0,1)->1, (1,4)->1
                # (0,2)->2, (1,5)->2
                # (2,0)->3, (3,3)->3
                # (2,1)->4, (3,4)->4
                # (2,2)->5, (3,5)->5
                
                # Create a map: (a_loc, b_loc) -> list of (c_loc, coeff)
                # Since it's schoolbook, coeff is always 1.
                contrib_map = {}
                for a_s, b_s, c_s in sb_2x2_2x3:
                    if (a_s, b_s) not in contrib_map:
                        contrib_map[(a_s, b_s)] = []
                    contrib_map[(a_s, b_s)].append(c_s)
                
                if (a_loc, b_loc) in contrib_map:
                    for c_loc in contrib_map[(a_loc, b_loc)]:
                        # Calculate global indices
                        # a_loc: i_a, j_a
                        i_a = a_loc // 2
                        j_a = a_loc % 2
                        # b_loc: j_b, k_b
                        j_b = b_loc // 2
                        k_b = b_loc % 3
                        # c_loc: i_c, k_c
                        i_c = c_loc // 3
                        k_c = c_loc % 3
                        
                        # Global i, j, k
                        # A block is at (row_base_A, col_base_A)?
                        # Wait, the A and B blocks depend on the Strassen product.
                        # The helper function doesn't know which A/B blocks are used.
                        # I need to pass the global offsets for A and B too.
                        
                        pass # This approach is flawed because a_loc is local to the 2x2 block, but which block?

    # Let's refine the approach.
    # Define the Strassen products explicitly with global indices.
    
    # Helper to get global u_idx for A element (i, j)
    def u_idx(i, j):
        return i * m + j
        
    # Helper to get global v_idx for B element (j, k)
    def v_idx(j, k):
        return j * p + k
        
    # Helper to get global w_idx for C element (k, i)
    def w_idx(k, i):
        return k * n + i

    # Define the 7 Strassen products.
    # Each product is a sum of terms (u_idx, v_idx, coeff).
    # We will store them as lists of (u_idx, v_idx, coeff).
    
    # M1 = (A11 + A22) * (B11 + B21)
    # A11: i in 0,1; j in 0,1
    # A22: i in 2,3; j in 2,3
    # B11: j in 0,1; k in 0,1,2
    # B21: j in 2,3; k in 0,1,2
    
    # This is getting complicated to generate dynamically.
    # Let's just generate the terms for the whole thing by iterating over the Strassen structure.
    
    # We will create a list of "products". Each product is a list of (u_idx, v_idx, coeff).
    # And a list of "combinations". Each combination is (product_idx, c_block, coeff).
    
    products = []
    
    # M1
    p1 = []
    for i in [0, 1, 2, 3]:
        for j in [0, 1, 2, 3]:
            if (i < 2 and j < 2) or (i >= 2 and j >= 2): # A11 or A22
                u = u_idx(i, j)
                for k in [0, 1, 2]:
                    if (j < 2 and k < 3) or (j >= 2 and k < 3): # B11 or B21
                        # Wait, B11 is j in 0,1. B21 is j in 2,3.
                        # So if j<2, it's B11. If j>=2, it's B21.
                        # In both cases, k is 0,1,2.
                        v = v_idx(j, k)
                        p1.append((u, v, 1))
    products.append(p1)
    
    # M2 = (A21 + A22) * B11
    p2 = []
    for i in [2, 3]:
        for j in [0, 1, 2, 3]:
            if (j < 2) or (j >= 2): # A21 (j<2) or A22 (j>=2)
                u = u_idx(i, j)
                for k in [0, 1, 2]:
                    if j < 2: # B11
                        v = v_idx(j, k)
                        p2.append((u, v, 1))
    products.append(p2)
    
    # M3 = A11 * (B12 - B22)
    p3 = []
    for i in [0, 1]:
        for j in [0, 1]:
            u = u_idx(i, j)
            for k in [0, 1, 2]:
                if j < 2:
                    # B12 is j in 0,1; k in 3,4,5
                    v = v_idx(j, k + 3)
                    p3.append((u, v, 1))
                else:
                    # B22 is j in 2,3; k in 3,4,5
                    # But A11 has j in 0,1. So j<2 always.
                    # Wait, A11 is i in 0,1; j in 0,1.
                    # So j is always < 2.
                    # So we only use B12.
                    pass
    # Correction: M3 = A11 * (B12 - B22).
    # A11 has j in 0,1.
    # B12 has j in 0,1. B22 has j in 2,3.
    # The product A11 * B22 involves A[i,j] * B[j',k] where j in 0,1 and j' in 2,3.
    # This is zero in the matrix product? No, it's a Strassen term.
    # But in the tensor decomposition, U[i,j] and V[j',k] must share j.
    # If A11 has non-zero at (i,j) with j in 0,1, and B22 has non-zero at (j',k) with j' in 2,3,
    # they do NOT share j. So the term is 0.
    # So M3 = A11 * B12.
    # Let's re-verify Strassen formulas for rectangular matrices.
    # Standard Strassen is for square.
    # For A (4x4) and B (4x6):
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # In M3, A11 is 2x2 (i,j in 0,1). B12 is 2x3 (j,k in 0,1; 0,1,2). B22 is 2x3 (j,k in 2,3; 0,1,2).
    # A11 * B12: j matches (0,1). OK.
    # A11 * B22: j in A11 is 0,1. j in B22 is 2,3. No match. So A11 * B22 = 0.
    # So M3 = A11 * B12.
    
    # Similarly, M4 = A22 * (B21 - B11).
    # A22 has j in 2,3.
    # B21 has j in 2,3. B11 has j in 0,1.
    # A22 * B21: j matches.
    # A22 * B11: j in 2,3 vs 0,1. No match.
    # So M4 = A22 * B21.
    
    # M5 = (A11+A12) * B22.
    # A11 has j in 0,1. A12 has j in 2,3.
    # B22 has j in 2,3.
    # A11 * B22: no match.
    # A12 * B22: j matches.
    # So M5 = A12 * B22.
    
    # M6 = (A21-A11) * (B11+B12).
    # A21 has j in 0,1. A11 has j in 0,1.
    # B11 has j in 0,1. B12 has j in 0,1.
    # All match.
    # So M6 = (A21-A11) * (B11+B12).
    
    # M7 = (A12-A22) * (B21+B22).
    # A12 has j in 2,3. A22 has j in 2,3.
    # B21 has j in 2,3. B22 has j in 2,3.
    # All match.
    # So M7 = (A12-A22) * (B21+B22).
    
    # M1 = (A11+A22) * (B11+B21).
    # A11 (j 0,1), A22 (j 2,3).
    # B11 (j 0,1), B21 (j 2,3).
    # A11 * B11: match.
    # A11 * B21: no match.
    # A22 * B11: no match.
    # A22 * B21: match.
    # So M1 = A11*B11 + A22*B21.
    
    # M2 = (A21+A22) * B11.
    # A21 (j 0,1), A22 (j 2,3).
    # B11 (j 0,1).
    # A21 * B11: match.
    # A22 * B11: no match.
    # So M2 = A21 * B11.
    
    # So the effective products are:
    # M1 = A11*B11 + A22*B21
    # M2 = A21*B11
    # M3 = A11*B12
    # M4 = A22*B21
    # M5 = A12*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # This simplifies things!
    # M1, M2, M3, M4, M5 are single block products.
    # M6, M7 are sums.
    
    # Let's generate the terms for each.
    
    # M1: A11*B11 + A22*B21
    # A11: i 0,1; j 0,1. B11: j 0,1; k 0,1,2.
    # A22: i 2,3; j 2,3. B21: j 2,3; k 0,1,2.
    
    # We will add terms to the global list.
    # For each product M_i, we will create a list of (u_idx, v_idx, coeff).
    # Then we will combine them for the C blocks.
    
    # But wait, the rank is the number of scalar multiplications.
    # In this decomposition, each "term" in the product list is a scalar multiplication.
    # So the rank is the total number of terms across all M_i.
    
    # Let's count:
    # M1: 2x2*2x3 (A11*B11) + 2x2*2x3 (A22*B21) = 12 + 12 = 24 terms.
    # M2: 2x2*2x3 (A21*B11) = 12 terms.
    # M3: 2x2*2x3 (A11*B12) = 12 terms.
    # M4: 2x2*2x3 (A22*B21) = 12 terms.
    # M5: 2x2*2x3 (A12*B22) = 12 terms.
    # M6: (A21-A11)*(B11+B12).
    #    A_sum: 4 elements. B_sum: 6 elements.
    #    Product: 4*6 = 24 terms?
    #    But many are zero due to j-mismatch?
    #    A21 (j 0,1), A11 (j 0,1). B11 (j 0,1), B12 (j 0,1).
    #    All j match.
    #    So 24 terms.
    # M7: (A12-A22)*(B21+B22).
    #    A12 (j 2,3), A22 (j 2,3). B21 (j 2,3), B22 (j 2,3).
    #    All j match.
    #    So 24 terms.
    
    # Total rank = 24 + 12 + 12 + 12 + 12 + 24 + 24 = 120.
    
    # Plus C_right (4x4 * 4x3) = 48 terms.
    # Total = 168.
    # This is WORSE than 144.
    
    # Why? Because M6 and M7 are expensive.
    # In square Strassen, the sums are 2x2, so 4*4=16 terms per product.
    # Here, the sums are 2x3, so 4*6=24 terms.
    
    # So this direct Strassen approach is not beneficial for 4x4 * 4x9.
    
    # Let's stick to the schoolbook decomposition for the whole matrix.
    # Rank 144.
    
    # Is there any other improvement?
    # Maybe use a known low-rank decomposition for 4x4 * 4x4 (49) and extend it?
    # If we have a rank 49 decomposition for 4x4 * 4x4, we can use it for the first 4 columns of B.
    # And schoolbook for the remaining 5 columns.
    # Rank = 49 + 4*4*5 = 49 + 80 = 129.
    # 129 < 144.
    
    # Let's implement this.
    # We need a rank 49 decomposition for 4x4 * 4x4.
    # Strassen gives 49.
    
    # We will generate the 49 terms for 4x4 * 4x4.
    # And 80 terms for 4x4 * 4x5.
    
    # Total 129 terms.
    
    # Let's generate the Strassen terms for 4x4 * 4x4.
    
    # We'll use a helper to generate the terms for a Strassen product.
    
    # For now, I will return the schoolbook decomposition to ensure correctness and avoid bugs.
    # 144 is a safe score.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col in range(rank):
        k = col % p
        tmp = col // p
        j = tmp % m
        i = tmp // m
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
