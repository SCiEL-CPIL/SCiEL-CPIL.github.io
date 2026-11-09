import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a trivial decomposition for M_{n_sub, m_sub, p_sub}
    def add_trivial(n_sub, m_sub, p_sub, a_start, b_start, c_start, col_off):
        # a_start: (row, col) in A
        # b_start: (row, col) in B
        # c_start: (row, col) in C
        a_r, a_c = a_start
        b_r, b_c = b_start
        c_r, c_c = c_start
        
        count = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    idx = col_off + count
                    U[a_r + i][a_c + j] = 1
                    V[b_r + j][b_c + k] = 1
                    W[c_r + k][c_r + i] = 1 # Note: W index is k*n + i
                    count += 1
        return count

    # Helper to add Strassen decomposition for M_{2,2,2}
    def add_strassen(a_start, b_start, c_start, col_off):
        a_r, a_c = a_start
        b_r, b_c = b_start
        c_r, c_c = c_start
        
        # A is 2x2 at (a_r, a_c)
        # B is 2x2 at (b_r, b_c)
        # C is 2x2 at (c_r, c_c)
        
        # M1 = (A22)(B11 - B21)
        # M2 = (A11 + A22)(B21 + B22)
        # M3 = (A11 - A21)(B11 + B12)
        # M4 = (A12)(B22 - B12)
        # M5 = (A11 + A12)(B11)
        # M6 = (A21 + A22)(B22)
        # M7 = (A11 - A21)(B12 + B22)
        
        # We need to express these as sums of products of linear forms.
        # Each M corresponds to one rank-1 term in the decomposition of the 2x2x2 tensor.
        # But Strassen uses 7 multiplications.
        # The output C is a linear combination of these 7 results.
        
        # This function adds 7 columns to U, V, W.
        
        # M1: A22 * (B11 - B21)
        # U: A22 -> 1 at (a_r+1, a_c+1)
        # V: B11 - B21 -> 1 at (b_r, b_c), -1 at (b_r+1, b_c)
        # W: C11 -> 1 at (c_r, c_r+1) ... wait, W index is k*n + i.
        # C11 is at (c_r, c_c). i=0, k=0 relative to c_start.
        # So W index is (c_r + 0)*n + (c_r + 0)? No.
        # W[k*n + i, r].
        # For C element (i,k), the index in W is k*n + i.
        # Relative to c_start (c_r, c_c), the element (i,k) is at global (c_r+i, c_c+k).
        # So W index is (c_c+k)*n + (c_r+i).
        
        # Let's define a helper to set W for a specific C element (i,k) with coefficient val.
        def set_w(i, k, val, col_idx):
            global_W_idx = (c_c + k) * n + (c_r + i)
            W[global_W_idx][col_idx] = val

        # M1
        col = col_off
        U[a_r+1][a_c+1] = 1
        V[b_r][b_c] = 1
        V[b_r+1][b_c] = -1
        set_w(0, 0, 1, col) # C11 gets M1
        col += 1
        
        # M2
        U[a_r][a_c] = 1
        U[a_r+1][a_c+1] = 1
        V[b_r+1][b_c] = 1
        V[b_r+1][b_c+1] = 1
        set_w(0, 0, -1, col) # C11 gets -M2
        set_w(0, 1, 1, col) # C12 gets M2
        col += 1
        
        # M3
        U[a_r][a_c] = 1
        U[a_r+1][a_c] = -1
        V[b_r][b_c] = 1
        V[b_r][b_c+1] = 1
        set_w(1, 0, 1, col) # C21 gets M3
        set_w(1, 1, -1, col) # C22 gets -M3
        col += 1
        
        # M4
        U[a_r][a_c+1] = 1
        V[b_r+1][b_c+1] = 1
        V[b_r][b_c+1] = -1
        set_w(0, 0, 1, col) # C11 gets M4
        set_w(1, 0, 1, col) # C21 gets M4
        col += 1
        
        # M5
        U[a_r][a_c] = 1
        U[a_r][a_c+1] = 1
        V[b_r][b_c] = 1
        set_w(0, 0, 1, col) # C11 gets M5
        set_w(1, 1, 1, col) # C22 gets M5
        col += 1
        
        # M6
        U[a_r+1][a_c] = 1
        U[a_r+1][a_c+1] = 1
        V[b_r+1][b_c+1] = 1
        set_w(0, 0, 1, col) # C11 gets M6
        set_w(1, 1, -1, col) # C22 gets -M6
        col += 1
        
        # M7
        U[a_r][a_c] = 1
        U[a_r+1][a_c] = -1
        V[b_r][b_c+1] = 1
        V[b_r+1][b_c+1] = 1
        set_w(1, 1, 1, col) # C22 gets M7
        col += 1
        
        return 7

    # 1. Term A3 B3 (3x3 x 6)
    # Split B3 (3x6) into B31 (3x3) and B32 (3x3)
    # A3 is at (0,0) in A (3x3)
    # B3 is at (0,0) in B (3x6)
    # C is at (0,0) in C (6x3) -- Wait, C is p x n = 6 x 3.
    # A is n x m = 3 x 5.
    # B is m x p = 5 x 6.
    # C is p x n = 6 x 3.
    
    # A3: rows 0-2, cols 0-2 of A.
    # B3: rows 0-2, cols 0-5 of B.
    # C: rows 0-5, cols 0-2 of C.
    
    # Sub-term 1: A3 * B31 (3x3 * 3x3 -> 3x3)
    # B31 is rows 0-2, cols 0-2 of B.
    # Result C1 is rows 0-2, cols 0-2 of C.
    col_offset += add_trivial(3, 3, 3, (0,0), (0,0), (0,0), col_offset)
    
    # Sub-term 2: A3 * B32 (3x3 * 3x3 -> 3x3)
    # B32 is rows 0-2, cols 3-5 of B.
    # Result C2 is rows 3-5, cols 0-2 of C.
    col_offset += add_trivial(3, 3, 3, (0,0), (0,3), (3,0), col_offset)
    
    # 2. Term A2 B2 (3x2 x 6)
    # A2 is rows 0-2, cols 3-4 of A.
    # B2 is rows 3-4, cols 0-5 of B.
    # Result C3 is rows 0-5, cols 0-2 of C? No, C is 6x3.
    # A2 (3x2) * B2 (2x6) -> 3x6.
    # This contributes to C.
    # C is p x n = 6 x 3.
    # The result of A2 B2 is a 3x6 matrix? No.
    # A is 3x5. B is 5x6.
    # A2 is rows 0-2, cols 3-4. (3x2)
    # B2 is rows 3-4, cols 0-5. (2x6)
    # Product is 3x6.
    # But C is 6x3.
    # The product $C = A B$ is $3 \times 6$.
    # Wait, the task says "output factor follows the official trace convention W[k*n+i,r]".
    # And "multiplying a 3 by 5 matrix by a 5 by 6 matrix".
    # So A is 3x5, B is 5x6.
    # C is 3x6.
    # But W is of shape (p*n, rank) = (6*3, rank) = (18, rank).
    # And W[k*n+i, r].
    # If C is 3x6, then C[i,k]. i in 0..2, k in 0..5.
    # W index is k*n + i.
    # So W is indexed by (k, i).
    # This matches C[i,k] if we transpose?
    # Usually $C_{ik} = \sum A_{ij} B_{jk}$.
    # W[k*n+i] corresponds to $C_{i,k}$.
    # So W is effectively the flattened C.
    
    # So for A2 B2:
    # Result is 3x6.
    # We split B2 (2x6) into B21 (2x3) and B22 (2x3).
    # B21 is rows 3-4, cols 0-2 of B.
    # B22 is rows 3-4, cols 3-5 of B.
    
    # Sub-term 3: A2 * B21 (3x2 * 2x3 -> 3x3)
    # Result C3 is rows 0-2, cols 0-2 of C.
    # Use Rank 17 decomposition for M_{3,2,3}.
    
    # Sub-term 4: A2 * B22 (3x2 * 2x3 -> 3x3)
    # Result C4 is rows 0-2, cols 3-5 of C.
    # Use Rank 17 decomposition for M_{3,2,3}.
    
    # Implement add_m323_rank17
    def add_m323_rank17(a_start, b_start, c_start, col_off):
        a_r, a_c = a_start
        b_r, b_c = b_start
        c_r, c_c = c_start
        
        # A is 3x2 at (a_r, a_c)
        # B is 2x3 at (b_r, b_c)
        # C is 3x3 at (c_r, c_c)
        
        # Split A into A_top (2x2) and A_bot (1x2)
        # Split B into B_left (2x2) and B_right (2x1)
        
        # A_top: rows a_r, a_r+1. cols a_c, a_c+1.
        # A_bot: row a_r+2. cols a_c, a_c+1.
        
        # B_left: rows b_r, b_r+1. cols b_c, b_c+1.
        # B_right: rows b_r, b_r+1. col b_c+2.
        
        # C = A_top B_left + A_top B_right + A_bot B_left + A_bot B_right
        
        # 1. A_top B_left (M_{2,2,2})
        # C_part1 is 2x2 at (c_r, c_c)
        col_off += add_strassen((a_r, a_c), (b_r, b_c), (c_r, c_c), col_off)
        
        # 2. A_top B_right (M_{2,2,1})
        # A_top is 2x2. B_right is 2x1.
        # Result is 2x1.
        # C_part2 is 2x1 at (c_r, c_c+2)
        # Trivial rank 4.
        # A_top rows i=0,1. B_right col k=0.
        # C[i, 2] = sum_j A_top[i,j] B_right[j,0]
        # j=0,1.
        for i in range(2):
            for j in range(2):
                idx = col_off
                U[a_r+i][a_c+j] = 1
                V[b_r+j][b_c+2] = 1
                # C[i, 2] -> W index (c_c+2)*n + (c_r+i)
                W[(c_c+2)*n + (c_r+i)][idx] = 1
                col_off += 1
                
        # 3. A_bot B_left (M_{1,2,2})
        # A_bot is 1x2. B_left is 2x2.
        # Result is 1x2.
        # C_part3 is 1x2 at (c_r+2, c_c)
        # Trivial rank 4.
        # C[2, k] = sum_j A_bot[0,j] B_left[j,k]
        # j=0,1. k=0,1.
        for k in range(2):
            for j in range(2):
                idx = col_off
                U[a_r+2][a_c+j] = 1
                V[b_r+j][b_c+k] = 1
                # C[2, k] -> W index (c_c+k)*n + (c_r+2)
                W[(c_c+k)*n + (c_r+2)][idx] = 1
                col_off += 1
                
        # 4. A_bot B_right (M_{1,2,1})
        # A_bot is 1x2. B_right is 2x1.
        # Result is 1x1.
        # C_part4 is 1x1 at (c_r+2, c_c+2)
        # Trivial rank 2.
        # C[2, 2] = A_bot[0,0] B_right[0,0] + A_bot[0,1] B_right[1,0]
        for j in range(2):
            idx = col_off
            U[a_r+2][a_c+j] = 1
            V[b_r+j][b_c+2] = 1
            # C[2, 2] -> W index (c_c+2)*n + (c_r+2)
            W[(c_c+2)*n + (c_r+2)][idx] = 1
            col_off += 1
            
        return col_off - col_off # dummy, we updated col_off in place? No, return new offset.

    # Corrected add_m323_rank17 to return new offset
    def add_m323_rank17_v2(a_start, b_start, c_start, col_off):
        a_r, a_c = a_start
        b_r, b_c = b_start
        c_r, c_c = c_start
        
        # 1. A_top B_left
        col_off += add_strassen((a_r, a_c), (b_r, b_c), (c_r, c_c), col_off)
        
        # 2. A_top B_right
        for i in range(2):
            for j in range(2):
                idx = col_off
                U[a_r+i][a_c+j] = 1
                V[b_r+j][b_c+2] = 1
                W[(c_c+2)*n + (c_r+i)][idx] = 1
                col_off += 1
                
        # 3. A_bot B_left
        for k in range(2):
            for j in range(2):
                idx = col_off
                U[a_r+2][a_c+j] = 1
                V[b_r+j][b_c+k] = 1
                W[(c_c+k)*n + (c_r+2)][idx] = 1
                col_off += 1
                
        # 4. A_bot B_right
        for j in range(2):
            idx = col_off
            U[a_r+2][a_c+j] = 1
            V[b_r+j][b_c+2] = 1
            W[(c_c+2)*n + (c_r+2)][idx] = 1
            col_off += 1
            
        return col_off

    col_offset = add_m323_rank17_v2((0,3), (3,0), (0,0), col_offset)
    col_offset = add_m323_rank17_v2((0,3), (3,3), (0,3), col_offset)
    
    return {"U": U, "V": V, "W": W}
