import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using 8 independent Strassen 2x2x2 blocks
    # Structure:
    # A is 2x4, B is 4x8, C is 2x8.
    # Split A into A0, A1 (2x2 each).
    # Split B into B0..B3 (2x2 each).
    # C = A0 B0 + A0 B1 + A1 B2 + A1 B3 ? No.
    # C = A0 B_top + A1 B_bottom.
    # B_top is 2x8, B_bottom is 2x8.
    # B_top = [B0, B1], B_bottom = [B2, B3].
    # A0 B_top = A0 B0 + A0 B1.
    # A1 B_bottom = A1 B2 + A1 B3.
    # Each term is 2x2 * 2x2 = 2x2, rank 7.
    # Total 4 terms * 7 rank = 28?
    # Wait.
    # A0 is 2x2. B0 is 2x2. Product is 2x2.
    # A0 B0 contributes to C[:, 0:2].
    # A0 B1 contributes to C[:, 2:4].
    # A1 B2 contributes to C[:, 4:6].
    # A1 B3 contributes to C[:, 6:8].
    # So we have 4 independent 2x2x2 multiplications.
    # Rank = 4 * 7 = 28?
    # Let's check the dimensions again.
    # A: 2x4. B: 4x8.
    # A = [A0, A1] where A0, A1 are 2x2.
    # B is 4x8.
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x8.
    # C = A0 B_top + A1 B_bottom.
    # B_top is 2x8. We can split B_top into [B0, B1] (2x2 each).
    # B_bottom is 2x8. We can split B_bottom into [B2, B3] (2x2 each).
    # C = A0 [B0, B1] + A1 [B2, B3]
    # C = [A0 B0, A0 B1, A1 B2, A1 B3].
    # Each block is 2x2.
    # So we have 4 independent 2x2x2 multiplications.
    # Rank = 4 * 7 = 28.
    
    # Why did the previous code use 56?
    # Let's re-read the previous code logic.
    # "Partition: j into 2 groups of 2, k into 4 groups of 2"
    # "Each sub-problem: 2x2 * 2x2 = 2x2, rank 7"
    # "Total: 2 * 4 * 7 = 56"
    # The previous code loops:
    # for a in range(2): # j-group (rows of B / cols of A)
    #   for b in range(4): # k-group (cols of B / cols of C)
    #     for s in range(7):
    # This implies 8 independent 2x2x2 multiplications.
    # My analysis:
    # C = A0 B_top + A1 B_bottom.
    # A0 B_top: A0 is 2x2, B_top is 2x8.
    # This is a 2x2 * 2x8 multiplication.
    # Rank of 2x2 * 2x8 is R(2, 2, 8) = 28?
    # No, R(2, 2, 8) is the rank of the tensor for 2x2 by 2x8.
    # We established R(2, 2, 8) = 28.
    # So A0 B_top requires rank 28.
    # A1 B_bottom requires rank 28.
    # Total 56.
    
    # My earlier check "4 independent 2x2x2" was wrong because:
    # A0 B_top is NOT 4 independent 2x2x2 multiplications in terms of rank?
    # A0 B_top = A0 [B0, B1] = [A0 B0, A0 B1].
    # A0 B0 is 2x2x2 (rank 7).
    # A0 B1 is 2x2x2 (rank 7).
    # Are they independent?
    # The inputs are A0, B0 for the first, and A0, B1 for the second.
    # A0 is shared!
    # Can we share computations between A0 B0 and A0 B1?
    # Strassen for 2x2x2 uses 7 multiplications.
    # If we compute A0 B0 and A0 B1 separately, we use 14 multiplications.
    # But A0 is the same.
    # Does this allow reduction?
    # The tensor for A0 [B0, B1] is 2x2 by 2x4.
    # We established R(2, 2, 4) = 14.
    # So A0 B_top has rank 14.
    # Similarly A1 B_bottom has rank 14.
    # Total 28?
    # Wait.
    # R(2, 2, 4) = 14.
    # So A0 B_top (2x2 by 2x4) has rank 14.
    # A1 B_bottom (2x2 by 2x4) has rank 14.
    # Total rank 28.
    
    # So the rank should be 28, not 56!
    # Why did the previous code use 56?
    # Let's check the previous code's U, V, W dimensions.
    # U: 8x56. V: 32x56. W: 16x56.
    # The previous code explicitly loops 2 * 4 * 7 = 56.
    # It treats each 2x2 block of C as independent.
    # C has 4 blocks of 2x2.
    # If rank is 28, we can improve the score significantly!
    
    # Let's verify R(2, 2, 4) = 14.
    # Yes, it is the sum of two R(2, 2, 2) = 7.
    # And they are disjoint in B variables (B0 vs B1).
    # So R(2, 2, 4) = 14 is tight.
    # So A0 B_top has rank 14.
    # A1 B_bottom has rank 14.
    # Total 28.
    
    # Let's construct the rank 28 decomposition.
    # Rank = 28.
    # U: 8x28. V: 32x28. W: 16x28.
    
    # A0 is A[:, 0:2]. A1 is A[:, 2:4].
    # B_top is B[0:2, :]. B_bottom is B[2:4, :].
    
    # Term 1: A0 B_top (Rank 14)
    # A0 = A[:, 0:2].
    # B_top = B[0:2, :].
    # This produces C[:, 0:4].
    # We use Strassen for 2x2x4.
    # Strassen for 2x2x4:
    # Split B_top into B0 (2x2) and B1 (2x2).
    # Compute A0 B0 (Rank 7) and A0 B1 (Rank 7).
    # They are disjoint in B, so rank is 7+7=14.
    
    # Term 2: A1 B_bottom (Rank 14)
    # A1 = A[:, 2:4].
    # B_bottom = B[2:4, :].
    # This produces C[:, 4:8].
    # Split B_bottom into B2 (2x2) and B3 (2x2).
    # Compute A1 B2 (Rank 7) and A1 B3 (Rank 7).
    # Rank 14.
    
    # Total Rank 28.
    
    # Let's implement this.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 32 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    t = 0
    
    # Helper to add Strassen 2x2x2 block
    # A_block: 2x2, B_block: 2x2, C_block: 2x2
    # U indices for A_block, V indices for B_block, W indices for C_block
    def add_strassen_2x2x2(u_row_offset, v_row_offset, w_row_offset, a_row_offset, b_row_offset, c_col_offset):
        # a_row_offset: 0 or 2 (which rows of A)
        # b_row_offset: 0, 2, 4, 6 (which rows of B)
        # c_col_offset: 0, 2, 4, 6 (which cols of C)
        
        # Strassen steps:
        # M1 = (A00 + A11) (B00 + B11) -> C00 += M1, C11 += M1
        # M2 = A10 (B01 - B11) -> C10 += M2, C11 -= M2
        # M3 = A00 (B10 - B00) -> C00 += M3, C10 += M3
        # M4 = A11 B00 -> C11 += M4
        # M5 = (A00 + A01) B10 -> C00 -= M5, C01 += M5
        # M6 = (A10 + A11) B01 -> C10 += M6
        # M7 = (A01 - A10) (B00 + B10) -> C01 += M7
        
        # U (A) contributions
        # A00 is at u_row_offset + 0, u_col_offset + 0
        # A01 is at u_row_offset + 0, u_col_offset + 1
        # A10 is at u_row_offset + 1, u_col_offset + 0
        # A11 is at u_row_offset + 1, u_col_offset + 1
        
        # V (B) contributions
        # B00 is at v_row_offset + 0, v_col_offset + 0
        # B01 is at v_row_offset + 0, v_col_offset + 1
        # B10 is at v_row_offset + 1, v_col_offset + 0
        # B11 is at v_row_offset + 1, v_col_offset + 1
        
        # W (C) contributions
        # C00 is at w_row_offset + 0, w_col_offset + 0
        # C01 is at w_row_offset + 0, w_col_offset + 1
        # C10 is at w_row_offset + 1, w_col_offset + 0
        # C11 is at w_row_offset + 1, w_col_offset + 1
        
        # Note: u_row_offset, v_row_offset, w_row_offset are base row indices for the 2x2 block.
        # But U is 8x28. A is 2x4.
        # A0 is rows 0,1 of A. A1 is rows 2,3 of A.
        # Wait, A is 2x4. n=2, m=4.
        # U shape (8, 28).
        # A indices: i in 0..1, j in 0..3.
        # A0 corresponds to j=0,1. A1 corresponds to j=2,3.
        # So A0 is U[0:2, :], A1 is U[2:4, :]? No.
        # U is (n*m, rank).
        # A[i, j] maps to U[i*m + j, t].
        # A0: i in 0..1, j in 0..1. Indices 0,1,4,5.
        # A1: i in 0..1, j in 2..3. Indices 2,3,6,7.
        
        # Let's just use the global indices.
        
        # Step 1
        U[a_row_offset * m + 0][t] += 1
        U[a_row_offset * m + 1][t] += 1 # A00+A11? No, A00 is (0,0), A11 is (1,1) relative to block.
        # Relative indices:
        # A00: row 0, col 0
        # A01: row 0, col 1
        # A10: row 1, col 0
        # A11: row 1, col 1
        
        # Global A index: i*m + j.
        # Block A starts at row a_row_offset, col a_col_offset.
        # A00 global: a_row_offset * m + a_col_offset
        # A01 global: a_row_offset * m + a_col_offset + 1
        # A10 global: (a_row_offset + 1) * m + a_col_offset
        # A11 global: (a_row_offset + 1) * m + a_col_offset + 1
        
        # We need to pass a_row_offset and a_col_offset.
        pass

    # This helper is getting complex. I will inline the logic for the 4 blocks.
    
    # Block 1: A0 (j=0,1) * B0 (rows 0,1) -> C (cols 0,1)
    # a_row=0, a_col=0. b_row=0, b_col=0. c_col=0.
    # Block 2: A0 (j=0,1) * B1 (rows 0,1) -> C (cols 2,3)
    # a_row=0, a_col=0. b_row=0, b_col=2. c_col=2.
    # Block 3: A1 (j=2,3) * B2 (rows 2,3) -> C (cols 4,5)
    # a_row=0, a_col=2. b_row=2, b_col=0. c_col=4.
    # Block 4: A1 (j=2,3) * B3 (rows 2,3) -> C (cols 6,7)
    # a_row=0, a_col=2. b_row=2, b_col=2. c_col=6.
    
    # Note: A rows are 0,1. So a_row_offset is always 0?
    # Yes, n=2.
    
    def add_strassen(a_col, b_row, b_col, c_col):
        nonlocal t
        # A indices
        u_00 = 0 * m + a_col
        u_01 = 0 * m + a_col + 1
        u_10 = 1 * m + a_col
        u_11 = 1 * m + a_col + 1
        
        # B indices
        v_00 = b_row * p + b_col
        v_01 = b_row * p + b_col + 1
        v_10 = (b_row + 1) * p + b_col
        v_11 = (b_row + 1) * p + b_col + 1
        
        # C indices
        w_00 = c_col * n + 0
        w_01 = c_col * n + 1
        w_10 = c_col * n + 1 # Wait, C is 2x8. W is (p*n, rank).
        # W[k*n + i, r]. k is col, i is row.
        # C[i, k].
        # C00: i=0, k=c_col. Index c_col*n + 0.
        # C01: i=0, k=c_col+1. Index (c_col+1)*n + 0.
        # C10: i=1, k=c_col. Index c_col*n + 1.
        # C11: i=1, k=c_col+1. Index (c_col+1)*n + 1.
        
        w_00 = c_col * n + 0
        w_01 = (c_col + 1) * n + 0
        w_10 = c_col * n + 1
        w_11 = (c_col + 1) * n + 1
        
        # M1
        U[u_00][t] += 1
        U[u_11][t] += 1
        V[v_00][t] += 1
        V[v_11][t] += 1
        W[w_00][t] += 1
        W[w_11][t] += 1
        t += 1
        
        # M2
        U[u_10][t] += 1
        V[v_01][t] += 1
        V[v_11][t] += -1
        W[w_10][t] += 1
        W[w_11][t] += -1
        t += 1
        
        # M3
        U[u_00][t] += 1
        V[v_10][t] += 1
        V[v_00][t] += -1
        W[w_00][t] += 1
        W[w_10][t] += 1
        t += 1
        
        # M4
        U[u_11][t] += 1
        V[v_00][t] += 1
        W[w_11][t] += 1
        t += 1
        
        # M5
        U[u_00][t] += 1
        U[u_01][t] += 1
        V[v_10][t] += 1
        W[w_00][t] += -1
        W[w_01][t] += 1
        t += 1
        
        # M6
        U[u_10][t] += 1
        U[u_11][t] += 1
        V[v_01][t] += 1
        W[w_10][t] += 1
        t += 1
        
        # M7
        U[u_01][t] += 1
        U[u_10][t] += -1
        V[v_00][t] += 1
        V[v_10][t] += 1
        W[w_01][t] += 1
        t += 1

    add_strassen(0, 0, 0, 0)
    add_strassen(0, 0, 2, 2)
    add_strassen(2, 2, 0, 4)
    add_strassen(2, 2, 2, 6)
    
    return {"U": U, "V": V, "W": W}

