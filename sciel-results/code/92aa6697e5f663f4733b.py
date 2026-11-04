"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _build_strassen_3x3x4():
    """
    Returns (U, V, W) for 3x3x4 matrix multiplication.
    U: shape (9, 16)
    V: shape (12, 16)
    W: shape (12, 16)
    Convention:
      A is 3x3, B is 3x4, C = A B is 3x4.
      U maps A elements (row-major) to rank indices.
      V maps B elements (row-major) to rank indices.
      W maps C elements (row-major, W[k*3+i, c]) to rank indices.
    """
    rank = 16
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    
    # Decomposition strategy:
    # We can view 3x3x4 as 3x4x4 + adjustments, but simpler:
    # Use block structure.
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    # B = [[B11, B12, B13, B14],
    #      [B21, B22, B23, B24],
    #      [B31, B32, B33, B34]]
    
    # Let's use a known rank-16 algorithm for 3x3x4?
    # Schoolbook rank is 36. Strassen-based might be lower.
    # Actually, the rank of 3x3x4 is known to be 16? 
    # Let's check: Rank of 3x3x3 is 16. 3x3x4 can be embedded?
    # A 3x3x4 tensor can be decomposed.
    # Let's try to construct it explicitly.
    
    # Split B into two 3x2 blocks: B_left, B_right.
    # C_left = A * B_left (3x2)
    # C_right = A * B_right (3x2)
    # We need rank for 3x3x2 + 3x3x2.
    # Rank of 3x3x2 is 9? Schoolbook 3*3*2=18.
    # Using Strassen for 2x2x2 parts?
    # Let's use a simpler approach: Just implement a known good rank.
    # If I can't find a compact explicit formula, I will use a generic Strassen-like block algorithm.
    
    # Actually, let's look at the 3x4x8 problem directly.
    # Strategy: 3x4x8 = 3x4x4 + 3x4x4.
    # If we find rank r for 3x4x4, total rank 2r.
    # Schoolbook 3x4x4 is 48. Rank 24 would give 48.
    # Can we do better than 48?
    # Rank of 3x4x4 is known to be 24?
    # Let's check: Rank of 2x2x2 is 7.
    # 3x4x4 can be decomposed into blocks.
    
    # Let's try to construct a rank-24 algorithm for 3x4x4.
    # A (3x4), B (4x4).
    # Split A into rows: A1 (1x4), A2 (2x4).
    # Split B into cols: B1 (4x2), B2 (4x2).
    # C1 = A1 * B1 (1x2), C2 = A1 * B2 (1x2)
    # C3 = A2 * B1 (2x2), C4 = A2 * B2 (2x2)
    
    # Rank of 1x4x2: Schoolbook 8. Can be reduced?
    # 1x4x2 tensor. A is row vector, B is 4x2.
    # This is just dot products. Rank is 2? No, 4*2=8 terms.
    # Each C_ij = sum_k A_k B_kj.
    # This is a bilinear form. Rank is min(4, 2) * 2? No.
    # The tensor T[i,j,k] = A_k B_kj.
    # This is the sum of 4 rank-1 tensors? No, k is summed.
    # It's 4 separate multiplications A_k * B_k1 and A_k * B_k2.
    # So 8 scalar mults. Can we reduce?
    # A=[a,b,c,d], B=[[b11,b12],[b21,b22],[b31,b32],[b41,b42]]
    # c1 = a*b11+b*b21+c*b31+d*b41
    # c2 = a*b12+b*b22+c*b32+d*b42
    # This is 1x4x2. Rank is 4?
    # We can compute (a+b)(b11+b21) etc?
    # (a+b)(b11+b21) = ab11+ab21+bb11+bb21.
    # We need ab11, bb21.
    # This seems hard to reduce below 4 for the 1x4 part?
    # Actually, rank of 1xm x 1 is m. Rank of 1xm xn is m*n?
    # No, rank of 1xm xn tensor is m*n?
    # T_{i,j,k} = A_k B_kj.
    # Flatten A (size m), B (size m*n).
    # It's a linear map from A to C for each column of B?
    # No, B is also input.
    # It's the tensor for multiplying a row vector (1xm) by a matrix (mxn).
    # This is just the dot product for each column.
    # Rank is mn?
    # Let's check 1x2x2.
    # a1*b11+a2*b21, a1*b12+a2*b22.
    # Rank 4?
    # (a1+a2)(b11+b21) = a1b11+a1b21+a2b11+a2b21.
    # (a1-a2)(b11-b21) = a1b11-a1b21-a2b11+a2b21.
    # Sum: 2a1b11 + 2a2b21.
    # We can recover a1b11 and a2b21.
    # Similarly for the other column.
    # So we need 4 multiplications for 1x2x2?
    # (a1+a2)(b11+b21), (a1-a2)(b11-b21), (a1+a2)(b12+b22), (a1-a2)(b12-b22).
    # From first two: a1b11, a2b21.
    # From last two: a1b12, a2b22.
    # So rank is 4. Schoolbook is 4. No gain.
    # So 1x4x2 is rank 8.
    
    # Back to 3x4x4:
    # C1, C2 (1x2 each) -> rank 8 + 8 = 16.
    # C3, C4 (2x2 each) -> rank 7 + 7 = 14 (Strassen).
    # Total rank 16 + 14 = 30.
    # So 3x4x4 rank <= 30.
    # Then 3x4x8 rank <= 60.
    # Schoolbook 3x4x8 is 96.
    # 60 is better than 96.
    
    # Can we do better on 3x4x4?
    # What if we split A into 3 rows of 1x4?
    # 3 * (1x4x2) + (1x4x2)? No.
    # A (3x4) * B (4x4).
    # Split B into 2 blocks of 4x2.
    # C_left (3x2), C_right (3x2).
    # Rank of 3x4x2?
    # Split A into 3 rows.
    # Each row is 1x4x2 (rank 8).
    # Total 24.
    # Can we combine?
    # Maybe not easily.
    # So 30 seems like a good target for 3x4x4.
    
    # Let's implement the 30-rank algorithm for 3x4x4.
    # And use it twice for 3x4x8 (rank 60).
    
    # Wait, is there a better way for 3x4x2?
    # A (3x4), B (4x2).
    # C = AB (3x2).
    # Rank 3x4x2.
    # Schoolbook 24.
    # Using Strassen on 2x2 blocks?
    # Split A into A1 (2x4), A2 (1x4).
    # Split B into B1 (4x1), B2 (4x1).
    # C11 = A1 B1 (2x1), C12 = A1 B2 (2x1)
    # C21 = A2 B1 (1x1), C22 = A2 B2 (1x1)
    # Rank 2x4x1:
    # Split A1 into rows? 1x4x1 (rank 4).
    # Two rows -> 8.
    # Rank 1x4x1: 4.
    # Total 16?
    # Can we reduce 2x4x1?
    # A (2x4), B (4x1).
    # c1 = a11b1+a12b2+a13b3+a14b4
    # c2 = a21b1+a22b2+a23b3+a24b4
    # Rank 8?
    # (a11+a21)(b1) ... no.
    # This is just 8 scalar mults.
    # So 3x4x2 rank is likely 24.
    
    # So 3x4x4 rank 30 seems solid.
    # Total 60.
    
    # Let's write the code for 3x4x4 rank 30.
    
    def _build_3x4x4():
        rank = 30
        U = [[0]*rank for _ in range(12)] # 3*4
        V = [[0]*rank for _ in range(16)] # 4*4
        W = [[0]*rank for _ in range(12)] # 4*3
        
        col = 0
        # Part 1: A_row0 (1x4) * B_left (4x2) -> C (1x2). Rank 8.
        # B_left is B[:, 0:2].
        # A_row0 is A[0, :].
        # Terms: A[0,j] * B[j, k] for j in 0..3, k in 0..1.
        for j in range(4):
            for k in range(2):
                U[0*4 + j][col] = 1
                V[j*4 + k][col] = 1
                W[k*3 + 0][col] = 1 # W index: col k, row i=0. W[k*3+i].
                col += 1
        
        # Part 2: A_row0 (1x4) * B_right (4x2) -> C (1x2). Rank 8.
        # B_right is B[:, 2:4].
        for j in range(4):
            for k in range(2, 4):
                U[0*4 + j][col] = 1
                V[j*4 + k][col] = 1
                W[k*3 + 0][col] = 1
                col += 1
        
        # Part 3: A_bottom (2x4) * B_left (4x2) -> C (2x2). Rank 14 (Strassen).
        # A_bottom is A[1:3, :]. Rows 1,2.
        # B_left is B[:, 0:2]. Cols 0,1.
        # C is rows 1,2 and cols 0,1.
        # We map A_bottom to 2x2 blocks? No, A is 2x4, B is 4x2.
        # We need to handle the 4 width.
        # Split A_bottom into A_bl (2x2), A_br (2x2).
        # Split B_left into B_tl (2x2), B_bl (2x2).
        # C_tl (1,0), C_tr (1,1), C_bl (2,0), C_br (2,1).
        
        # A_bl: rows 1,2; cols 0,1.
        # A_br: rows 1,2; cols 2,3.
        # B_tl: rows 0,1; cols 0,1.
        # B_bl: rows 2,3; cols 0,1.
        
        # C_10 = A_bl B_tl + A_br B_bl
        # C_11 = A_bl B_tl_col1? No, B_left is 4x2.
        # B_left cols are 0 and 1.
        # B_tl is B[0:2, 0:2] -> 2x2.
        # B_bl is B[2:4, 0:2] -> 2x2.
        # C_10 (row 1, col 0) = sum_j A[1,j] B[j,0]
        # = sum_{j=0,1} A[1,j] B[j,0] + sum_{j=2,3} A[1,j] B[j,0]
        # = (A_bl row 0) . (B_tl col 0) + (A_br row 0) . (B_bl col 0)
        
        # This is getting complicated to map to standard 2x2x2 Strassen.
        # Standard Strassen is for 2x2 * 2x2.
        # Here we have 2x4 * 4x2.
        # We can view it as (2x2 | 2x2) * [[2x2], [2x2]] = (2x2)(2x2) + (2x2)(2x2).
        # This is a sum of two 2x2x2 tensors.
        # Rank is 7+7=14 if we do them separately.
        # Can we merge?
        # T = A1 B1 + A2 B2.
        # A1, A2 are 2x2. B1, B2 are 2x2.
        # This is not a single matrix mult.
        # So 14 is correct.
        
        # Let's implement the two separate Strassen blocks for Part 3 and Part 4.
        
        def add_strassen_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, offset):
            # a_row_start, a_col_start: top-left of A block (2x2)
            # b_row_start, b_col_start: top-left of B block (2x2)
            # c_row_start, c_col_start: top-left of C block (2x2)
            # Returns new col offset.
            
            # Strassen formulas for 2x2x2
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B12-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # We need to map the coefficients to U, V, W.
            # Each M is a product of (linear combo of A) and (linear combo of B).
            
            def add_term(u_coeffs, v_coeffs, w_coeff, col_idx, sign):
                # u_coeffs: dict {(row, col): coeff}
                # v_coeffs: dict {(row, col): coeff}
                # w_coeff: dict {(row, col): coeff}
                for (ar, ac), c in u_coeffs.items():
                    U[(ar + a_row_start)*4 + (ac + a_col_start)][col_idx] += sign * c
                for (br, bc), c in v_coeffs.items():
                    V[(br + b_row_start)*4 + (bc + b_col_start)][col_idx] += sign * c
                for (cr, cc), c in w_coeff.items():
                    W[(cc + c_col_start)*3 + (cr + c_row_start)][col_idx] += sign * c
            
            # M1
            col += 1
            add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1, (0,1):0, (1,0):0}, col-1, 1)
            # Wait, W coefficients for C11: M1 + M4 - M5 + M7.
            # So for M1, W[0,0] += 1, W[1,1] += 0? No, M1 contributes to C11 and C22.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # Let's redo this carefully.
            
            # M1
            col_idx = col
            col += 1
            add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, col_idx, 1)
            
            # M2
            col_idx = col
            col += 1
            add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}, col_idx, 1)
            
            # M3
            col_idx = col
            col += 1
            add_term({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}, col_idx, 1)
            
            # M4
            col_idx = col
            col += 1
            add_term({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}, col_idx, 1)
            
            # M5
            col_idx = col
            col += 1
            add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):1}, col_idx, 1)
            
            # M6
            col_idx = col
            col += 1
            add_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}, col_idx, 1)
            
            # M7
            col_idx = col
            col += 1
            add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}, col_idx, 1)
            
            return col
        
        # Part 3: A_bottom (rows 1,2) * B_left (cols 0,1) -> C (rows 1,2, cols 0,1)
        # Split A into A_bl (cols 0,1) and A_br (cols 2,3).
        # Split B into B_tl (rows 0,1) and B_bl (rows 2,3).
        
        # Term 1: A_bl * B_tl
        col = add_strassen_2x2x2(1, 0, 0, 0, 1, 0, col)
        
        # Term 2: A_br * B_bl
        # A_br is rows 1,2; cols 2,3.
        # B_bl is rows 2,3; cols 0,1.
        # C is same target: rows 1,2; cols 0,1.
        # The Strassen algorithm computes C = A B.
        # We just add the contributions.
        # So we call add_strassen_2x2x2 with the new block coords.
        col = add_strassen_2x2x2(1, 2, 2, 0, 1, 0, col)
        
        # Part 4: A_bottom (rows 1,2) * B_right (cols 2,3) -> C (rows 1,2, cols 2,3)
        # Term 1: A_bl * B_tr
        # A_bl: rows 1,2; cols 0,1.
        # B_tr: rows 0,1; cols 2,3.
        # C: rows 1,2; cols 2,3.
        col = add_strassen_2x2x2(1, 0, 0, 2, 1, 2, col)
        
        # Term 2: A_br * B_br
        # A_br: rows 1,2; cols 2,3.
        # B_br: rows 2,3; cols 2,3.
        # C: rows 1,2; cols 2,3.
        col = add_strassen_2x2x2(1, 2, 2, 2, 1, 2, col)
        
        assert col == rank, f"Rank mismatch: {col} vs {rank}"
        return U, V, W
    
    return _build_3x4x4()

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # We build 3x4x4 rank 30 twice.
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # C is 3x8. Split into C1 (3x4) and C2 (3x4).
    
    U_base, V_base, W_base = _build_strassen_3x3x4() # Actually returns 3x4x4
    
    rank = 60
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block 1: B[:, 0:4]
    for i in range(n*m):
        for c in range(30):
            U[i][c] = U_base[i][c]
    
    for i in range(m*p):
        if i < m*4: # First 4 cols
            for c in range(30):
                V[i][c] = V_base[i][c]
        # else 0
    
    for i in range(p*n):
        # W index: k*n + r.
        # C1 is cols 0:4.
        # W_base has indices for k in 0..3.
        # We need to map k in 0..3 to global k in 0..3.
        for k in range(4):
            for r in range(3):
                local_idx = k*3 + r
                global_idx = k*3 + r
                for c in range(30):
                    W[global_idx][c] = W_base[local_idx][c]
    
    # Block 2: B[:, 4:8]
    # Offset in V: 4. Offset in W: 4.
    for i in range(n*m):
        for c in range(30):
            U[i][c+30] = U_base[i][c]
    
    for i in range(m*p):
        if i >= m*4 and i < m*8: # Last 4 cols
            local_i = i - m*4
            for c in range(30):
                V[i][c+30] = V_base[local_i][c]
    
    for i in range(p*n):
        # C2 is cols 4:8.
        # k in 4..7.
        for k in range(4, 8):
            for r in range(3):
                local_k = k - 4
                local_idx = local_k*3 + r
                global_idx = k*3 + r
                for c in range(30):
                    W[global_idx][c+30] = W_base[local_idx][c]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
