import time
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses 8 independent 2x2x2 Strassen sub-problems.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Each sub-problem is a 2x2 * 2x2 -> 2x2 multiplication.
    # We partition the 4x8 matrix B into 4 blocks of 2x8? No.
    # Actually: A is 2x4, B is 4x8.
    # We can think of the multiplication as C = A * B.
    # Let's partition the columns of A into 2 groups of 2, and the rows of B into 2 groups of 2? No, B has 4 rows.
    # Let's partition the 4 inner dimension into 4 groups of 1? No.
    
    # The known trick for 2x4 * 4x8:
    # View the 4x8 matrix as four 1x8 rows. But Strassen is for 2x2 blocks.
    # Let's partition A into two 2x2 blocks: A = [A11, A12] where A11 is 2x2 (cols 0,1), A12 is 2x2 (cols 2,3).
    # Let's partition B into four 2x4 blocks? No, B is 4x8.
    # B rows 0,1 form B_top (2x8). B rows 2,3 form B_bot (2x8).
    # C = A11 * B_top + A12 * B_bot.
    # This is a sum of two 2x2 * 2x8 products.
    # We can further split B_top into two 2x4 blocks: B_top = [B11, B12] (cols 0-3, 4-7).
    # B_bot = [B21, B22] (cols 0-3, 4-7).
    # So C = A11 * [B11, B12] + A12 * [B21, B22]
    #      = [A11*B11 + A12*B21, A11*B12 + A12*B22]
    # The left half of C is 2x4: C_left = A11*B11 + A12*B21.
    # The right half of C is 2x4: C_right = A11*B12 + A12*B22.
    
    # Now, C_left is a 2x4 matrix. It is the sum of two 2x2 * 2x2 products.
    # Let X = A11*B11 and Y = A12*B21. Then C_left = X + Y.
    # We can compute X and Y using Strassen? No, Strassen computes a single 2x2 product with 7 mults.
    # Here we have two separate 2x2 products. That would be 7+7=14 mults for C_left.
    # Similarly for C_right: 7+7=14 mults.
    # Total 28 mults? No, that's if we just use Strassen independently.
    # But the rank 56 solution implies 56 mults total.
    # 56 / 8 = 7. So there are 8 independent Strassen blocks.
    
    # Let's re-examine the structure.
    # A is 2x4. B is 4x8.
    # Maybe we split the 4 inner dimension into 4 parts of size 1? No.
    # Maybe we split A into 2x2 and 2x2. And B into 4 2x4 blocks?
    # Let A = [A_L, A_R] (2x2 each).
    # Let B be partitioned into 4 blocks of size 2x4:
    # B = [B1; B2; B3; B4] where each Bi is 2x4.
    # C = A_L * [B1; B2] + A_R * [B3; B4]
    #   = A_L*B1 + A_L*B2 + A_R*B3 + A_R*B4.
    # Each term is 2x2 * 2x4 -> 2x4.
    # We can split each 2x4 product into two 2x2 * 2x2 products.
    # A_L*B1 = [A_L*B1_left, A_L*B1_right] (each 2x2 * 2x2).
    # So we have 4 (A_B) pairs, each split into 2 (2x2 * 2x2) products.
    # Total 8 (2x2 * 2x2) products.
    # Each uses Strassen (rank 7).
    # Total rank = 8 * 7 = 56.
    
    # The 8 products are:
    # 1. A_L * B1_left  (A_L is cols 0,1 of A; B1_left is rows 0,1 cols 0,3 of B? No, rows 0,1 cols 0,1)
    #    Wait, B1 is rows 0,1. B1_left is cols 0,1.
    # 2. A_L * B1_right (cols 2,3)
    # 3. A_L * B2_left  (rows 2,3 cols 0,1)
    # 4. A_L * B2_right (rows 2,3 cols 2,3)
    # 5. A_R * B3_left  (A_R is cols 2,3 of A; B3 is rows 0,1? No, B is partitioned by rows 0,1; 2,3; ... wait)
    #    Let's clarify B partitioning.
    #    B is 4x8.
    #    B1 = B[0:2, 0:4]  (2x4)
    #    B2 = B[0:2, 4:8]  (2x4)
    #    B3 = B[2:4, 0:4]  (2x4)
    #    B4 = B[2:4, 4:8]  (2x4)
    #    This doesn't match A_L * [B1; B2] logic.
    #    A_L is 2x2 (cols 0,1). It multiplies rows 0,1 of B.
    #    So A_L * B[0:2, :] which is 2x8.
    #    Let B_top = B[0:2, :] (2x8).
    #    A_L * B_top = A_L * [B_top_left, B_top_right] where B_top_left = B[0:2, 0:4] (2x4) and B_top_right = B[0:2, 4:8] (2x4).
    #    Each 2x4 product splits into two 2x2 * 2x2.
    #    So A_L contributes 4 Strassen blocks.
    #    A_R (cols 2,3) multiplies rows 2,3 of B.
    #    B_bot = B[2:4, :] (2x8).
    #    A_R * B_bot splits into 4 Strassen blocks.
    #    Total 8 blocks.
    
    # Block definitions:
    # Let's index the 8 blocks as 0..7.
    # Block 0: A_L * B[0:2, 0:2]  -> C[0:2, 0:2]
    # Block 1: A_L * B[0:2, 2:4]  -> C[0:2, 2:4]
    # Block 2: A_L * B[0:2, 4:6]  -> C[0:2, 4:6]
    # Block 3: A_L * B[0:2, 6:8]  -> C[0:2, 6:8]
    # Block 4: A_R * B[2:4, 0:2]  -> C[0:2, 0:2]  (add to Block 0's result? No, C is 2x8. C[0:2, 0:2] gets contribution from Block 0 AND Block 4? No.)
    #    Wait. C = A * B.
    #    C[i,k] = sum_j A[i,j] B[j,k].
    #    C[0:2, 0:2] = A[0:2, 0:4] * B[0:4, 0:2].
    #    = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2].
    #    So yes, C[0:2, 0:2] is sum of Block 0 and Block 4 results.
    #    This means the "products" are not independent in the output, but the *multiplications* are independent scalar operations that are then summed.
    #    In tensor decomposition, each rank-1 term is a scalar multiplication.
    #    Strassen gives 7 scalar multiplications for a 2x2 product.
    #    The additions are handled by the U, V, W coefficients.
    
    # Strassen formulas for X = P * Q (2x2):
    # M1 = (P00 + P11) * (Q00 + Q11)
    # M2 = (P10 + P11) * Q00
    # M3 = P00 * (Q01 - Q11)
    # M4 = P11 * (Q10 - Q00)
    # M5 = (P00 + P01) * Q11
    # M6 = (P10 - P00) * (Q00 + Q01)
    # M7 = (P01 - P11) * (Q10 + Q11)
    # X00 = M1 + M4 - M5 + M7
    # X01 = M3 + M5
    # X10 = M2 + M4
    # X11 = M1 - M2 + M3 + M6
    
    # We need to construct U, V, W such that C = U * diag(rank) * V^T? No.
    # The tensor T[i,j,k] for C[i,k] = sum_j A[i,j] B[j,k] is decomposed as sum_r U[i,j,r] V[j,k,r] W[i,k,r].
    # Here, for each Strassen block, we have 7 terms.
    # For a block computing X = P * Q where P is 2x2 sub-matrix of A, Q is 2x2 sub-matrix of B, and X adds to a 2x2 sub-matrix of C.
    # Let the block be identified by (a_row, a_col, b_row, b_col) ranges.
    # Actually, let's just define the 8 blocks explicitly.
    
    # Block b (0..7):
    # b = 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # b = 1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # b = 2: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # b = 3: A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
    # b = 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # b = 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # b = 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # b = 7: A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8]
    
    # Wait, A is 2x4.
    # A_L = A[0:2, 0:2] (cols 0,1)
    # A_R = A[0:2, 2:4] (cols 2,3)
    # B_top = B[0:2, :] (rows 0,1)
    # B_bot = B[2:4, :] (rows 2,3)
    # B_top is 2x8. Split into 4 blocks of 2x2:
    # BT0 = B[0:2, 0:2], BT1 = B[0:2, 2:4], BT2 = B[0:2, 4:6], BT3 = B[0:2, 6:8]
    # B_bot is 2x8. Split into 4 blocks of 2x2:
    # BB0 = B[2:4, 0:2], BB1 = B[2:4, 2:4], BB2 = B[2:4, 4:6], BB3 = B[2:4, 6:8]
    
    # C[0:2, 0:2] = A_L * BT0 + A_R * BB0
    # C[0:2, 2:4] = A_L * BT1 + A_R * BB1
    # C[0:2, 4:6] = A_L * BT2 + A_R * BB2
    # C[0:2, 6:8] = A_L * BT3 + A_R * BB3
    
    # So we have 8 Strassen multiplications.
    # Let's create the U, V, W matrices.
    # Shape of U: (n*m, rank) = (2*4, 56) = (8, 56)
    # Shape of V: (m*p, rank) = (4*8, 56) = (32, 56)
    # Shape of W: (p*n, rank) = (8*2, 56) = (16, 56)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M1 = (P00 + P11) * (Q00 + Q11)
    # M2 = (P10 + P11) * Q00
    # M3 = P00 * (Q01 - Q11)
    # M4 = P11 * (Q10 - Q00)
    # M5 = (P00 + P01) * Q11
    # M6 = (P10 - P00) * (Q00 + Q01)
    # M7 = (P01 - P11) * (Q10 + Q11)
    
    # For a block with A indices (a_i0, a_i1, a_j0, a_j1) and B indices (b_i0, b_i1, b_j0, b_j1)
    # P is 2x2: P[0,0]=A[a_i0, a_j0], P[0,1]=A[a_i0, a_j1], P[1,0]=A[a_i1, a_j0], P[1,1]=A[a_i1, a_j1]
    # Q is 2x2: Q[0,0]=B[b_i0, b_j0], Q[0,1]=B[b_i0, b_j1], Q[1,0]=B[b_i1, b_j0], Q[1,1]=B[b_i1, b_j1]
    # Result X adds to C[a_i0:a_i2, b_j0:b_j2] where a_i2=a_i0+2, b_j2=b_j0+2.
    # C indices in W: W[k*n + i, r]. Here i is row of C, k is col of C.
    # So W index is col*n + row.
    
    strassen_defs = [
        # (P_coeffs, Q_coeffs, X_coeffs)
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,-1), (1,0,1)]), # X00, X01? No, let's be careful.
        # Let's use a helper function to generate the rows for a block.
    ]
    
    def add_strassen_block(a_row0, a_col0, b_row0, b_col0, block_idx):
        # a_row0, a_col0: top-left of A submatrix (2x2)
        # b_row0, b_col0: top-left of B submatrix (2x2)
        # C submatrix is at rows a_row0, a_row0+1 and cols b_col0, b_col0+1
        
        # P indices in A:
        # P00: A[a_row0, a_col0]      -> U index a_row0*m + a_col0
        # P01: A[a_row0, a_col0+1]    -> U index a_row0*m + a_col0+1
        # P10: A[a_row0+1, a_col0]    -> U index (a_row0+1)*m + a_col0
        # P11: A[a_row0+1, a_col0+1]  -> U index (a_row0+1)*m + a_col0+1
        
        # Q indices in B:
        # Q00: B[b_row0, b_col0]      -> V index b_row0*p + b_col0
        # Q01: B[b_row0, b_col0+1]    -> V index b_row0*p + b_col0+1
        # Q10: B[b_row0+1, b_col0]    -> V index (b_row0+1)*p + b_col0
        # Q11: B[b_row0+1, b_col0+1]  -> V index (b_row0+1)*p + b_col0+1
        
        # X indices in C (added to):
        # X00: C[a_row0, b_col0]      -> W index b_col0*n + a_row0
        # X01: C[a_row0, b_col0+1]    -> W index (b_col0+1)*n + a_row0
        # X10: C[a_row0+1, b_col0]    -> W index b_col0*n + (a_row0+1)
        # X11: C[a_row0+1, b_col0+1]  -> W index (b_col0+1)*n + (a_row0+1)
        
        # Define the 7 multiplications
        # Each is a list of (index, coeff) for U, V, and W contributions.
        
        # M1 = (P00 + P11) * (Q00 + Q11)
        # X00 += M1, X11 -= M1
        u1 = [(a_row0*m + a_col0, 1), ((a_row0+1)*m + a_col0+1, 1)]
        v1 = [(b_row0*p + b_col0, 1), ((b_row0+1)*p + b_col0+1, 1)]
        w1 = [(b_col0*n + a_row0, 1), ((b_col0+1)*n + (a_row0+1), -1)]
        
        # M2 = (P10 + P11) * Q00
        # X10 += M2, X11 -= M2
        u2 = [((a_row0+1)*m + a_col0, 1), ((a_row0+1)*m + a_col0+1, 1)]
        v2 = [(b_row0*p + b_col0, 1)]
        w2 = [(b_col0*n + (a_row0+1), 1), ((b_col0+1)*n + (a_row0+1), -1)]
        
        # M3 = P00 * (Q01 - Q11)
        # X01 += M3, X11 += M3
        u3 = [(a_row0*m + a_col0, 1)]
        v3 = [(b_row0*p + b_col0+1, 1), ((b_row0+1)*p + b_col0+1, -1)]
        w3 = [((b_col0+1)*n + a_row0, 1), ((b_col0+1)*n + (a_row0+1), 1)]
        
        # M4 = P11 * (Q10 - Q00)
        # X00 += M4, X10 += M4
        u4 = [((a_row0+1)*m + a_col0+1, 1)]
        v4 = [((b_row0+1)*p + b_col0, 1), (b_row0*p + b_col0, -1)]
        w4 = [(b_col0*n + a_row0, 1), (b_col0*n + (a_row0+1), 1)]
        
        # M5 = (P00 + P01) * Q11
        # X00 -= M5, X01 += M5
        u5 = [(a_row0*m + a_col0, 1), (a_row0*m + a_col0+1, 1)]
        v5 = [((b_row0+1)*p + b_col0+1, 1)]
        w5 = [(b_col0*n + a_row0, -1), ((b_col0+1)*n + a_row0, 1)]
        
        # M6 = (P10 - P00) * (Q00 + Q01)
        # X11 += M6
        u6 = [((a_row0+1)*m + a_col0, 1), (a_row0*m + a_col0, -1)]
        v6 = [(b_row0*p + b_col0, 1), (b_row0*p + b_col0+1, 1)]
        w6 = [((b_col0+1)*n + (a_row0+1), 1)]
        
        # M7 = (P01 - P11) * (Q10 + Q11)
        # X00 += M7
        u7 = [(a_row0*m + a_col0+1, 1), ((a_row0+1)*m + a_col0+1, -1)]
        v7 = [((b_row0+1)*p + b_col0, 1), ((b_row0+1)*p + b_col0+1, 1)]
        w7 = [(b_col0*n + a_row0, 1)]
        
        blocks = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7),
        ]
        
        for s, (u_c, v_c, w_c) in enumerate(blocks):
            col = block_idx * 7 + s
            for idx, coeff in u_c:
                U[idx][col] += coeff
            for idx, coeff in v_c:
                V[idx][col] += coeff
            for idx, coeff in w_c:
                W[idx][col] += coeff

    # Define the 8 blocks
    # A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
    # BT0..3 = B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    # BB0..3 = B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8]
    
    # Block 0: A_L * BT0 -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0)
    # Block 1: A_L * BT1 -> C[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 1)
    # Block 2: A_L * BT2 -> C[0:2, 4:6]
    add_strassen_block(0, 0, 0, 4, 2)
    # Block 3: A_L * BT3 -> C[0:2, 6:8]
    add_strassen_block(0, 0, 0, 6, 3)
    # Block 4: A_R * BB0 -> C[0:2, 0:2]
    add_strassen_block(0, 2, 2, 0, 4)
    # Block 5: A_R * BB1 -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2, 2, 5)
    # Block 6: A_R * BB2 -> C[0:2, 4:6]
    add_strassen_block(0, 2, 2, 4, 6)
    # Block 7: A_R * BB3 -> C[0:2, 6:8]
    add_strassen_block(0, 2, 2, 6, 7)
    
    return {"U": U, "V": V, "W": W}
