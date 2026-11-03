"""Rewritten program for 3x4x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy:
    # Use Strassen (rank 7) for 2x2x2 blocks.
    # A is 3x4, B is 4x8, C is 3x8.
    # Split A into rows: top 2 rows (3x2) and bottom 1 row (1x2) -> wait, A is 3x4.
    # Split A columns into 2 blocks of 2: A_left (3x2), A_right (3x2).
    # Split B rows into 2 blocks of 2: B_top (2x8), B_bottom (2x8).
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # Let's split further to use Strassen on 2x2 sub-blocks where possible.
    # A_left is 3x2. B_top is 2x8.
    # A_right is 3x2. B_bottom is 2x8.
    #
    # We can decompose each 3x2 * 2x8 product.
    # Let A_sub be 3x2, B_sub be 2x8.
    # C_sub = A_sub * B_sub.
    # C_sub[i, k] = A_sub[i, 0] * B_sub[0, k] + A_sub[i, 1] * B_sub[1, k].
    # This is a sum of two rank-1 tensors? No, it's a bilinear form.
    # For fixed i, k, it's a sum over j=0,1.
    # This is equivalent to 3x2 * 2x8.
    # We can use schoolbook: 3*2*8 = 48 terms? No, rank is 3*2*8? No.
    # Rank of 3x2x2 * 2x8?
    # Actually, we can view this as:
    # C = A * B.
    #
    # Let's try a different split to maximize Strassen usage.
    # A (3x4), B (4x8).
    # Pad A to 4x4 with zeros? Pad B to 4x8?
    # If we use 4x4x8:
    # Split 4x4x8 into two 4x4x4 blocks?
    # 4x4x4 rank is 49 (Winograd) or 4x4x2x2?
    # Strassen 4x4x4: 4 * 7 = 28? No, 4x4x4 is 2x2x2 of 2x2x2?
    # 4x4x4 = 2x2x2 blocks of size 2.
    # There are 2*2*2 = 8 blocks of 2x2x2.
    # Each 2x2x2 block uses rank 7.
    # Total rank = 8 * 7 = 56.
    # But we need 3x4x8.
    #
    # Let's use the 4x4x8 approach with Strassen on 2x2x2 blocks.
    # A is 3x4. Pad to 4x4 (add row of zeros).
    # B is 4x8.
    # C is 3x8. (We ignore the 4th row of the resulting 4x8 matrix).
    #
    # Decompose 4x4x8 into 8 blocks of 2x2x2?
    # A is 4x4, B is 4x8.
    # Split A into 2x2 blocks: A00, A01, A10, A11.
    # Split B into 2x2 blocks: B00, B01, B10, B11 (rows 0-1, 2-3; cols 0-1, 2-3, 4-5, 6-7).
    # Wait, B is 4x8. So B has 2 row-blocks of size 2, and 4 col-blocks of size 2.
    # A has 2 row-blocks of size 2, and 2 col-blocks of size 2.
    #
    # C[i, k] = sum_j A[i, j] B[j, k].
    #
    # Let's index blocks:
    # A_{r, c} where r in {0,1}, c in {0,1}. Size 2x2.
    # B_{r, k} where r in {0,1}, k in {0,1,2,3}. Size 2x2.
    # C_{r, k} where r in {0,1}, k in {0,1,2,3}. Size 2x2.
    #
    # C_{r, k} = sum_{c=0,1} A_{r, c} * B_{c, k}.
    #
    # For each fixed r, k, this is a 2x2 * 2x2 * 2x2 sum?
    # No, A_{r, c} is 2x2, B_{c, k} is 2x2.
    # The product A_{r, c} B_{c, k} is a 2x2 matrix.
    # So C_{r, k} = A_{r, 0} B_{0, k} + A_{r, 1} B_{1, k}.
    #
    # This is a sum of two 2x2x2 matrix multiplications.
    # Each 2x2x2 multiplication can be done with rank 7 (Strassen).
    # So for each (r, k) pair, we need 2 * 7 = 14 rank units.
    # Number of (r, k) pairs: r in {0,1}, k in {0,1,2,3} -> 2 * 4 = 8 pairs.
    # Total rank = 8 * 14 = 112.
    # This is worse than 88.
    #
    # Why? Because Strassen combines the 8 multiplications into 7.
    # Here, for each (r,k), we have 2 multiplications.
    # Can we combine them?
    # C_{r, k} = A_{r, 0} B_{0, k} + A_{r, 1} B_{1, k}.
    # This is like multiplying a 2x4 matrix (A_{r,0}, A_{r,1}) by a 4x2 matrix (B_{0,k}, B_{1,k})?
    # No, dimensions don't match for a single Strassen block.
    #
    # Let's go back to the 88 rank solution which was:
    # 2x2x8 blocks (Strassen x4) -> 28 each. Two such blocks -> 56.
    # 1x2x8 blocks (schoolbook) -> 16 each. Two such blocks -> 32.
    # Total 88.
    #
    # The current program implements this.
    # Is there a better way?
    #
    # What if we use Winograd 3x3x3?
    # Rank 23.
    #
    # Let's try to fit 3x4x8 into 3x3x3 blocks.
    #
    # I will stick with the 88 rank solution but clean up the code significantly.
    # The previous code had some redundant logic and comments.
    # I will rewrite it cleanly.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    def add_strassen_block(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c1, b_c2, c_r1, c_r2, c_c1, c_c2):
        """
        Adds Strassen decomposition for:
        A[a_r1:a_r2+1, a_c1:a_c2+1] (2x2)
        B[b_r1:b_r2+1, b_c1:b_c2+1] (2x2)
        C[c_r1:c_r2+1, c_c1:c_c2+1] (2x2)
        """
        u_a = a_r1 * m + a_c1
        u_b = a_r1 * m + a_c2
        u_c = a_r2 * m + a_c1
        u_d = a_r2 * m + a_c2
        
        v_e = b_r1 * p + b_c1
        v_f = b_r1 * p + b_c2
        v_g = b_r2 * p + b_c1
        v_h = b_r2 * p + b_c2
        
        w_c11 = c_c1 * n + c_r1
        w_c12 = c_c2 * n + c_r1
        w_c21 = c_c1 * n + c_r2
        w_c22 = c_c2 * n + c_r2
        
        # M1 = (a+d)(e+h)
        add_rank_unit({u_a: 1, u_d: 1}, {v_e: 1, v_h: 1}, {w_c11: 1, w_c22: 1})
        # M2 = (c+d)e
        add_rank_unit({u_c: 1, u_d: 1}, {v_e: 1}, {w_c21: 1, w_c22: -1})
        # M3 = a(f-h)
        add_rank_unit({u_a: 1}, {v_f: 1, v_h: -1}, {w_c12: 1, w_c22: 1})
        # M4 = d(g-e)
        add_rank_unit({u_d: 1}, {v_g: 1, v_e: -1}, {w_c11: 1, w_c21: 1})
        # M5 = (a+b)h
        add_rank_unit({u_a: 1, u_b: 1}, {v_h: 1}, {w_c11: -1, w_c12: 1})
        # M6 = (c-a)(e+f)
        add_rank_unit({u_c: 1, u_a: -1}, {v_e: 1, v_f: 1}, {w_c22: 1})
        # M7 = (b-d)(g+h)
        add_rank_unit({u_b: 1, u_d: -1}, {v_g: 1, v_h: 1}, {w_c11: 1})

    # Block 1: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
    # A rows 0,1. A cols 0,1.
    # B rows 0,1. B cols 0-7.
    # C rows 0,1. C cols 0-7.
    for k_start in range(0, 8, 2):
        add_strassen_block(0, 1, 0, 1, 0, 1, k_start, k_start+1, 0, 1, k_start, k_start+1)

    # Block 2: A[0:2, 2:4] * B[2:4, 0:8] -> C[0:2, 0:8]
    # A rows 0,1. A cols 2,3.
    # B rows 2,3. B cols 0-7.
    # C rows 0,1. C cols 0-7.
    for k_start in range(0, 8, 2):
        add_strassen_block(0, 1, 2, 3, 2, 3, k_start, k_start+1, 0, 1, k_start, k_start+1)

    # Block 3: A[2:3, 0:2] * B[0:2, 0:8] -> C[2:3, 0:8]
    # A row 2. A cols 0,1.
    # B rows 0,1. B cols 0-7.
    # C row 2. C cols 0-7.
    # Schoolbook: 1x2 * 2x8 -> 1x8.
    # C[2, k] = A[2,0]B[0,k] + A[2,1]B[1,k]
    for k in range(8):
        add_rank_unit({2*m + 0: 1}, {0*p + k: 1}, {k*n + 2: 1})
        add_rank_unit({2*m + 1: 1}, {1*p + k: 1}, {k*n + 2: 1})

    # Block 4: A[2:3, 2:4] * B[2:4, 0:8] -> C[2:3, 0:8]
    # A row 2. A cols 2,3.
    # B rows 2,3. B cols 0-7.
    # C row 2. C cols 0-7.
    # Schoolbook: 1x2 * 2x8 -> 1x8.
    # C[2, k] = A[2,2]B[2,k] + A[2,3]B[3,k]
    for k in range(8):
        add_rank_unit({2*m + 2: 1}, {2*p + k: 1}, {k*n + 2: 1})
        add_rank_unit({2*m + 3: 1}, {3*p + k: 1}, {k*n + 2: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
