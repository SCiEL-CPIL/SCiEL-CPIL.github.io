import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = 3, 3, 5
    
    # We will build the rank 43 decomposition.
    # Split A (n x m) into A_top (2 x 3) and A_bot (1 x 3).
    # Split B (m x p) into B_left (3 x 2) and B_right (3 x 3).
    
    # Block 1: A_top (2x3) * B_left (3x2) -> C_top_left (2x2)
    # Rank 11 (Strassen 7 + Schoolbook 4? No, M(2,3,2) <= 11)
    # M(2,3,2) decomposition:
    # Split B_left into two 3x1 columns.
    # Col 1: 2x3 * 3x1 -> Rank 6
    # Col 2: 2x3 * 3x1 -> Rank 6
    # Total 12? 
    # Wait, M(2,3,2) <= 11 via M(2,2,2)+M(2,1,2).
    # Let's use the 11 rank decomposition for 2x3 * 3x2.
    
    # Block 2: A_top (2x3) * B_right (3x3) -> C_top_right (2x3)
    # Split B_right into B_r1 (3x2) and B_r2 (3x1).
    # A_top * B_r1 -> 2x2 (Rank 11)
    # A_top * B_r2 -> 2x1 (Rank 6)
    # Total 17.
    
    # Block 3: A_bot (1x3) * B_left (3x2) -> C_bot_left (1x2)
    # Rank 6.
    
    # Block 4: A_bot (1x3) * B_right (3x3) -> C_bot_right (1x3)
    # Rank 9.
    
    # Total Rank 11 + 17 + 6 + 9 = 43.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        U.append(u_idx)
        V.append(v_idx)
        W.append(w_idx)
        
    # Indices in U: i*m + j. A is 3x3.
    # i in 0..2, j in 0..2.
    # A_top rows: i=0,1. A_bot row: i=2.
    # B_left cols: k=0,1. B_right cols: k=2,3,4.
    
    # --- Block 1: A_top (2x3) * B_left (3x2) ---
    # We need rank 11 for 2x3 * 3x2.
    # Decomposition of M(2,3,2) using M(2,2,2) and M(2,1,2).
    # Split B (3x2) into B1 (3x1, col 0) and B2 (3x1, col 1)? No, that gives 12.
    # Split A (2x3) into A1 (1x3) and A2 (1x3)? 
    # A1 * B -> 1x2 (Rank 6).
    # A2 * B -> 1x2 (Rank 6).
    # Total 12.
    
    # Standard M(2,3,2) rank 11:
    # Split A into A1 (1x2), A2 (1x1)? No.
    # Let's use the known decomposition for 2x2x2 (Strassen) on a 2x2 subpart.
    # A (2x3), B (3x2).
    # Let A = [A11 A12; A21 A22] where A11 is 1x1? No.
    # A = [[a11, a12, a13], [a21, a22, a23]]
    # B = [[b11, b12], [b21, b22], [b31, b32]]
    # 
    # We can view this as:
    # C = [ [a11, a12, a13] * B ]  (1x2) -> Rank 6
    #     [ [a21, a22, a23] * B ]  (1x2) -> Rank 6
    # Total 12.
    
    # Is 11 possible?
    # M(2,3,2) = 11?
    # Yes, M(2,3,2) <= M(2,2,2) + M(2,1,2) = 7 + 4 = 11.
    # How to construct?
    # Split B into B1 (3x1) and B2 (3x1).
    # This gave 12.
    # Split A into A1 (1x3) and A2 (1x3).
    # This gave 12.
    # 
    # Split B into B1 (3x2) ? No.
    # 
    # Let's use the explicit 11 rank decomposition.
    # Reference: "The rank of the tensor for 2x3x2 matrix multiplication is 11".
    # 
    # Actually, let's just use 12 for simplicity if 11 is hard to code quickly.
    # 12 * 4 blocks? No, only 1 block is 2x3x2.
    # Block 1: 12.
    # Block 2: A_top (2x3) * B_right (3x3).
    #   Split B_right into 3x2 and 3x1.
    #   2x3 * 3x2 -> 12.
    #   2x3 * 3x1 -> 6.
    #   Total 18.
    # Block 3: 1x3 * 3x2 -> 6.
    # Block 4: 1x3 * 3x3 -> 9.
    # Total: 12 + 18 + 6 + 9 = 45.
    # This is no better than schoolbook.
    
    # We MUST use Strassen (Rank 7) for 2x2 parts.
    
    # Revised Plan for Rank 43:
    # Block 1: A_top (2x3) * B_left (3x2).
    # Split A_top into A_t1 (1x3) and A_t2 (1x3).
    # This is bad.
    # 
    # Let's split A_top (2x3) into A11 (1x1), A12 (1x2)?
    # 
    # Let's stick to the 43 estimate which relied on M(2,3,2)=11.
    # If I can't implement 11, I might get 45.
    # 
    # Let's try to implement Strassen for the 2x2 core.
    # In Block 1 (2x3 * 3x2), consider the sub-product of first 2 cols of A and first 2 cols of B?
    # A_sub (2x2) * B_sub (2x2) -> Strassen (7).
    # Remaining terms:
    # A has 3rd col. B has no more cols.
    # Terms: A_col3 * B_col1 and A_col3 * B_col2.
    # A_col3 is 2x1. B_col1 is 1x2? No, B is 3x2.
    # 
    # Let's do this:
    # A = [A_2x2, A_2x1].
    # B = [B_2x2; B_1x2].
    # C = A_2x2 * B_2x2 + A_2x1 * B_1x2.
    # Term 1: 2x2 * 2x2. Rank 7.
    # Term 2: 2x1 * 1x2.
    # A_2x1 is vector u (2x1). B_1x2 is vector v^T (1x2).
    # u * v^T is outer product. Rank 4?
    # u = [u1, u2]^T. v = [v1, v2].
    # uv^T = [[u1v1, u1v2], [u2v1, u2v2]].
    # 4 multiplies.
    # Total 7 + 4 = 11.
    # YES! This is the rank 11 decomposition.
    
    # So, for any 2x3 * 3x2 product, we can use Rank 11.
    
    # Now, let's map this to the indices.
    
    # Block 1: A_top (rows 0,1) * B_left (cols 0,1).
    # A_top is 2x3. B_left is 3x2.
    # Split A_top into A11 (rows 0,1; cols 0,1) -> 2x2.
    # Split A_top into A12 (rows 0,1; col 2) -> 2x1.
    # Split B_left into B11 (rows 0,1; cols 0,1) -> 2x2.
    # Split B_left into B12 (row 2; cols 0,1) -> 1x2.
    # 
    # C1 = A11*B11 (Strassen, rank 7) + A12*B12 (Outer, rank 4).
    
    # Block 2: A_top (rows 0,1) * B_right (cols 2,3,4).
    # B_right is 3x3.
    # Split B_right into B21 (rows 0,1; cols 2,3) -> 2x2.
    # B22 (row 2; cols 2,3) -> 1x2.
    # B23 (rows 0,1,2; col 4) -> 3x1.
    # 
    # This is getting messy. Let's split B_right into 3x2 and 3x1.
    # B_r1 (cols 2,3) is 3x2.
    # B_r2 (col 4) is 3x1.
    # 
    # A_top * B_r1: 2x3 * 3x2. Rank 11 (as above).
    # A_top * B_r2: 2x3 * 3x1.
    #   Split A_top into 2x2 and 2x1?
    #   A11 (2x2) * B_r2_part1 (2x1) -> 2x1. Rank 4?
    #   A12 (2x1) * B_r2_part2 (1x1) -> 2x1. Rank 2?
    #   Total 6.
    # 
    # So Block 2 Rank = 11 + 6 = 17.
    
    # Block 3: A_bot (row 2) * B_left (cols 0,1).
    # 1x3 * 3x2.
    # Split A_bot into 1x2 and 1x1.
    # B_left into 2x2 and 1x2.
    # (1x2)*(2x2) -> 1x2. Rank 4?
    # (1x1)*(1x2) -> 1x2. Rank 2?
    # Total 6.
    
    # Block 4: A_bot (row 2) * B_right (cols 2,3,4).
    # 1x3 * 3x3.
    # Split B_right into 2x2, 1x2, 3x1?
    # (1x2)*(2x2) -> 1x2. Rank 4.
    # (1x1)*(1x2) -> 1x2. Rank 2.
    # (1x3)*(3x1) -> 1x1. Rank 3.
    # Total 9.
    
    # Total 11 + 17 + 6 + 9 = 43.
    
    # Now, I need to write the code for these specific decompositions.
    
    pass
