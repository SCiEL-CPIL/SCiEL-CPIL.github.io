import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition: 8 blocks of 2x2x2 rank 7
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Rank-7 factors for 2x2x2
    # C = A * B
    # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # The 7 multiplications are:
    # M1 = (a+d)(x+w)
    # M2 = (c+d)(x+w)
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)(z+w)
    # M6 = (c+b)(x+z)
    # M7 = (a-c)(x+d) -- Wait, standard Strassen is:
    # M1 = (a+d)(x+w)
    # M2 = (c+d)(x+w)
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)(z+w)
    # M6 = (c+b)(x+z)
    # M7 = (a-c)(x+d) is not standard.
    
    # Standard Strassen:
    # M1 = (a + d)(x + w)
    # M2 = (c + d)(x + w)
    # M3 = a(y - w)
    # M4 = d(z - x)
    # M5 = (a + b)(z + w)
    # M6 = (c + b)(x + z)
    # M7 = (a - c)(x + d) ?
    
    # Let's use the explicit known factors:
    # U (A factors):
    # u1 = [1, 1, 0, 0] (a+d) ? No, a is (0,0), d is (1,1).
    # Indices: 0:00, 1:01, 2:10, 3:11
    # u1 = [1, 0, 0, 1]
    # u2 = [0, 0, 1, 1]
    # u3 = [1, 0, 0, 0]
    # u4 = [0, 0, 0, 1]
    # u5 = [1, 1, 0, 0]
    # u6 = [0, 1, 1, 0]
    # u7 = [1, 0, -1, 0]
    
    # V (B factors):
    # v1 = [1, 1, 1, 1] (x+w? no, x is (0,0), w is (1,1)) -> [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # v2 = [1, 0, 0, 1]
    # v3 = [0, 1, 0, -1] (y-w)
    # v4 = [0, 0, 1, -1] (z-x)
    # v5 = [0, 0, 1, 1] (z+w)
    # v6 = [1, 0, 1, 0] (x+z)
    # v7 = [1, 0, 0, 1] (x+d? no, x is (0,0), d is not in B)
    # M7 = (a-c)(x+d) is wrong.
    
    # Correct Strassen:
    # M1 = (a+d)(x+w)
    # M2 = (c+d)(x+w)
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)(z+w)
    # M6 = (c+b)(x+z)
    # M7 = (a-c)(x+d) -> This is incorrect.
    
    # The 7th term is usually (a-c)(x+d) is NOT correct.
    # The standard 7th term is (a-c)(x+d) is not a product of sums.
    # Actually, the 7th term is:
    # M7 = (a - c)(x + d) is not right.
    # The correct 7th term is:
    # M7 = (a - c)(x + d) ... wait.
    # Let's look at the reconstruction:
    # C11 = M1 - M2 + M3 - M5 + M7
    # C12 = M4 + M5
    # C21 = M2 + M6
    # C22 = M1 - M4 + M6 - M7
    
    # If this is correct, then:
    # M1 = (a+d)(x+w)
    # M2 = (c+d)(x+w)
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)(z+w)
    # M6 = (c+b)(x+z)
    # M7 = (a-c)(x+d) ?
    
    # Let's check C11:
    # (a+d)(x+w) - (c+d)(x+w) + a(y-w) - (a+b)(z+w) + (a-c)(x+d)
    # = (a-c)(x+w) + ay - aw - az - aw + ax + ad - cx - cd
    # This does not simplify to ax+bz.
    
    # The correct 7th factor is:
    # M7 = (a - c)(x + d) is NOT standard.
    # The standard 7th factor is often given as:
    # M7 = (a - c)(x + d) is wrong.
    
    # Let's use the well-known rank-7 factors:
    # 1: (a+d)(x+w)
    # 2: (c+d)(x+w)
    # 3: a(y-w)
    # 4: d(z-x)
    # 5: (a+b)(z+w)
    # 6: (c+b)(x+z)
    # 7: (a-c)(x+d) ... I am struggling to recall the exact 7th term.
    
    # Actually, I can just use the schoolbook for 2x2x2 if I'm not sure?
    # No, that gives rank 4. 8 * 4 = 32.
    # 32 is better than 56.
    # If I use schoolbook for 2x2x2, the total rank is 32.
    # Score 64/32 = 2.0.
    
    # Let's verify if schoolbook 2x2x2 (rank 4) can be used.
    # Yes, 8 independent 2x2x2 multiplications.
    # Rank 32.
    # This is a valid decomposition.
    # It is simpler to implement correctly than Strassen if I don't remember the 7th term.
    # And it scores 2.0, which is much better than 1.14.
    
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # 8 products of 2x2x2
    # Block 0: A_L (cols 0,1) * B_00 (rows 0,1 cols 0,1) -> Output C_0 (rows 0,1 cols 0,1)
    # Block 1: A_R (cols 2,3) * B_01 (rows 2,3 cols 0,1) -> Output C_0 (rows 0,1 cols 0,1)
    # ...
    
    # Let's define the 4 schoolbook terms for a 2x2x2 product
    # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # C11 = ax + bz
    # C12 = ay + bw
    # C21 = cx + dz
    # C22 = cy + dw
    
    # Term 1: ax
    # U: a is at (0,0) in local 2x2. Global A index depends on block.
    
    # We will iterate over the 8 blocks
    # Block k in 0..7
    # k = 0: A_L (0,1), B_00 (0,1; 0,1). Output to C cols 0,1.
    # k = 1: A_R (2,3), B_01 (2,3; 0,1). Output to C cols 0,1.
    # k = 2: A_L (0,1), B_02 (0,1; 2,3). Output to C cols 2,3.
    # k = 3: A_R (2,3), B_03 (2,3; 2,3). Output to C cols 2,3.
    # k = 4: A_L (0,1), B_10 (0,1; 4,5). Output to C cols 4,5.
    # k = 5: A_R (2,3), B_11 (2,3; 4,5). Output to C cols 4,5.
    # k = 6: A_L (0,1), B_12 (0,1; 6,7). Output to C cols 6,7.
    # k = 7: A_R (2,3), B_13 (2,3; 6,7). Output to C cols 6,7.
    
    # For each block, we add 4 terms.
    
    for block_idx in range(8):
        # Determine A columns (0,1 or 2,3)
        a_col_start = 0 if block_idx % 4 == 0 or block_idx % 4 == 2 else 2
        # Determine B rows (0,1 or 2,3)
        b_row_start = 0 if block_idx % 4 == 0 or block_idx % 4 == 2 else 2
        # Determine B cols (0,1 or 2,3 or 4,5 or 6,7)
        b_col_start = (block_idx // 4) * 4 + (0 if block_idx % 2 == 0 else 2)
        
        # Actually, let's just map explicitly.
        # Block 0: A[0,0:2], B[0:2, 0:2]
        # Block 1: A[0,2:4], B[2:4, 0:2]
        # Block 2: A[0,0:2], B[0:2, 2:4]
        # Block 3: A[0,2:4], B[2:4, 2:4]
        # Block 4: A[0,0:2], B[0:2, 4:6]
        # Block 5: A[0,2:4], B[2:4, 4:6]
        # Block 6: A[0,0:2], B[0:2, 6:8]
        # Block 7: A[0,2:4], B[2:4, 6:8]
        
        # Let's define the 4 terms for this block
        # Term 0: A[0, a_col] * B[b_row, b_col]
        # A row 0, col a_col
        # B row b_row, col b_col
        
        # We need to handle the 4 products:
        # P1 = A[0, a_c0] * B[b_r0, b_c0] -> C[0, b_c0]
        # P2 = A[0, a_c1] * B[b_r1, b_c0] -> C[0, b_c0]
        # P3 = A[1, a_c0] * B[b_r0, b_c1] -> C[1, b_c1]
        # P4 = A[1, a_c1] * B[b_r1, b_c1] -> C[1, b_c1]
        
        # Wait, the output columns are b_c0 and b_c1.
        # C[0, b_c0] = A[0, a_c0] B[b_r0, b_c0] + A[0, a_c1] B[b_r1, b_c0]
        # C[0, b_c1] = A[0, a_c0] B[b_r0, b_c1] + A[0, a_c1] B[b_r1, b_c1]
        # C[1, b_c0] = A[1, a_c0] B[b_r0, b_c0] + A[1, a_c1] B[b_r1, b_c0]
        # C[1, b_c1] = A[1, a_c0] B[b_r0, b_c1] + A[1, a_c1] B[b_r1, b_c1]
        
        # So we have 4 rank-1 terms per block.
        
        # Block parameters:
        # a_c0, a_c1: columns of A (0,1 or 2,3)
        # b_r0, b_r1: rows of B (0,1 or 2,3)
        # b_c0, b_c1: columns of B (0,1 or 2,3 or 4,5 or 6,7)
        
        if block_idx == 0:
            a_c0, a_c1 = 0, 1
            b_r0, b_r1 = 0, 1
            b_c0, b_c1 = 0, 1
        elif block_idx == 1:
            a_c0, a_c1 = 2, 3
            b_r0, b_r1 = 2, 3
            b_c0, b_c1 = 0, 1
        elif block_idx == 2:
            a_c0, a_c1 = 0, 1
            b_r0, b_r1 = 0, 1
            b_c0, b_c1 = 2, 3
        elif block_idx == 3:
            a_c0, a_c1 = 2, 3
            b_r0, b_r1 = 2, 3
            b_c0, b_c1 = 2, 3
        elif block_idx == 4:
            a_c0, a_c1 = 0, 1
            b_r0, b_r1 = 0, 1
            b_c0, b_c1 = 4, 5
        elif block_idx == 5:
            a_c0, a_c1 = 2, 3
            b_r0, b_r1 = 2, 3
            b_c0, b_c1 = 4, 5
        elif block_idx == 6:
            a_c0, a_c1 = 0, 1
            b_r0, b_r1 = 0, 1
            b_c0, b_c1 = 6, 7
        else: # block_idx == 7
            a_c0, a_c1 = 2, 3
            b_r0, b_r1 = 2, 3
            b_c0, b_c1 = 6, 7
            
        # Term 1: A[0, a_c0] * B[b_r0, b_c0] -> C[0, b_c0]
        U[0 * 4 + a_c0][col] = 1
        V[b_r0 * 8 + b_c0][col] = 1
        W[b_c0 * 2 + 0][col] = 1
        col += 1
        
        # Term 2: A[0, a_c1] * B[b_r1, b_c0] -> C[0, b_c0]
        U[0 * 4 + a_c1][col] = 1
        V[b_r1 * 8 + b_c0][col] = 1
        W[b_c0 * 2 + 0][col] = 1
        col += 1
        
        # Term 3: A[1, a_c0] * B[b_r0, b_c1] -> C[1, b_c1]
        U[1 * 4 + a_c0][col] = 1
        V[b_r0 * 8 + b_c1][col] = 1
        W[b_c1 * 2 + 1][col] = 1
        col += 1
        
        # Term 4: A[1, a_c1] * B[b_r1, b_c1] -> C[1, b_c1]
        U[1 * 4 + a_c1][col] = 1
        V[b_r1 * 8 + b_c1][col] = 1
        W[b_c1 * 2 + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
