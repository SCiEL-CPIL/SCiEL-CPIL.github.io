import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2 (rows of A), m=4 (cols of A = rows of B), p=6 (cols of B)
    # Output C is n x p = 2 x 6, indexed as C[i][k]
    # Tensor: T[i, j, k] = A[i,j] * B[j,k]
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (12, rank), indexed W[k*n+i, r]
    
    # Use rank-3 decomposition for 2x3x3 matrix multiplication
    # A is 2x3, B is 3x3, C is 2x3
    # Then apply to two blocks: cols 0-2 of B and cols 3-5 of B
    # Total rank: 3 + 3 = 6? No wait...
    # Actually for 2x3x3: min rank is 6 (schoolbook) or can we do better?
    # The minimum rank for multiplying a 2x3 by 3x3 matrix is 6 (schoolbook).
    # But wait, there's a known rank-5 decomposition? Let me think...
    # Actually no, the tensor rank of 2x3x3 matrix multiplication is 6.
    # 
    # Let me reconsider. The current approach uses Strassen (rank 7) for 2x2x2 blocks.
    # We have 2x4 * 4x6. 
    # 
    # Alternative: partition B into two 4x3 blocks.
    # A (2x4) * B1 (4x3) gives C1 (2x3), rank of this tensor is 6 (schoolbook)
    # A (2x4) * B2 (4x3) gives C2 (2x3), rank of this tensor is 6 (schoolbook)
    # Total rank: 12
    # 
    # But we can do better by using the structure. Let me try a different partition.
    # 
    # Actually, let me try to use the Laderman's algorithm or other fast methods.
    # 
    # For now, let me implement a rank-12 decomposition using schoolbook on two 4x3 blocks.
    # This should be valid and give score 48/12 = 4.0
    # 
    # Wait, the seed is rank 48, so score = 48/12 = 4.0. That's a big improvement!
    # 
    # But can we do even better? Let me think about whether there's a rank < 12 decomposition.
    # 
    # The 2x4x6 tensor... Let me just go with the safe rank-12 first.
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition for two independent 2x3x3 multiplications
    # Block 1: A (2x4) * B_cols_0_2 (4x3) -> C_cols_0_2 (2x3)
    # Block 2: A (2x4) * B_cols_3_5 (4x3) -> C_cols_3_5 (2x3)
    
    # For each block, schoolbook uses 4*3 = 12 terms? No, 2*4*3 = 24 terms.
    # Wait, schoolbook for 2x4 * 4x3: sum over j=0..3 of A[i,j]*B[j,k] for each i,k
    # That's 2*3*4 = 24 scalar multiplications, so rank 24 per block, total 48.
    # 
    # I'm confused. Let me re-examine.
    # 
    # The tensor rank of the matrix multiplication tensor M_{n,m,p} is the minimum
    # number of rank-1 tensors needed. For schoolbook, it's n*m*p.
    # So for 2x4x6, schoolbook rank = 2*4*6 = 48. That's the seed.
    # 
    # Strassen for 2x2x2 has rank 7 instead of 8.
    # The current program partitions into blocks and uses Strassen, getting rank 42.
    # 
    # To do better than 42, I need a more clever decomposition.
    # 
    # Let me try: partition A into two 2x2 blocks, B into three 4x2 blocks.
    # A = [A1, A2] where A1, A2 are 2x2
    # B = [B1; B2; B3] where B1, B2, B3 are 4x2
    # C = A1*B1 + A2*B2 ... no that's not right.
    # 
    # C = A*B where A is 2x4, B is 4x6.
    # Partition columns of B: B = [B_left | B_right] where B_left is 4x3, B_right is 4x3
    # C = A*B_left + ... no, C = [A*B_left | A*B_right]
    # 
    # A*B_left: (2x4)*(4x3) -> 2x3, tensor rank 2*4*3 = 24 schoolbook
    # A*B_right: same, 24
    # Total: 48
    # 
    # With Strassen-like tricks on the 2x4x3 tensor... 
    # 
    # Actually, let me just carefully implement the rank-42 Strassen decomposition 
    # but with a different structure to see if I can get it to rank 40 or less.
    # 
    # For now, let me implement a clean rank-42 to ensure validity, then optimize.
    
    # Use the known Strassen-based approach:
    # Partition A into left (2x2) and right (2x2)
    # Partition B into top (2x6) and bottom (2x6)  
    # C_left = A_left * B_top, C_right = A_right * B_bot ... no
    # 
    # C = A*B
    # A = [A_L | A_R] (2x2 each)
    # B = [B_T; B_B] (2x6 each)
    # C = A_L * B_T + A_R * B_B
    # 
    # A_L * B_T: (2x2)*(2x6) -> 2x6, rank 2*2*6 = 24 schoolbook, or with Strassen...
    # This doesn't decompose nicely into 2x2x2 blocks.
    # 
    # The current program's approach:
    # It partitions into 6 blocks of 2x2 * 2x2, each handled with Strassen (rank 7)
    # 6 * 7 = 42
    # 
    # Let me verify the block structure makes sense:
    # A is 2x4, B is 4x6
    # Block (i,j) where i in {0,1} for A's col groups, j in {0,1,2} for B's col groups
    # A group 0: cols 0-1, A group 1: cols 2-3
    # B group 0: rows 0-1, cols 0-1
    # B group 1: rows 0-1, cols 2-3  
    # B group 2: rows 0-1, cols 4-5
    # B group 3: rows 2-3, cols 0-1
    # B group 4: rows 2-3, cols 2-3
    # B group 5: rows 2-3, cols 4-5
    # 
    # A group 0 * B groups 0,1,2 (top rows of B)
    # A group 1 * B groups 3,4,5 (bottom rows of B)
    # Each is 2x2 * 2x2, Strassen rank 7
    # Total: 6 * 7 = 42
    # 
    # This is correct. Let me just implement it cleanly.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (A_indices, B_indices, C_indices, signs)
        # Each term: list of (local_idx, coeff) for A, B, C
        # Local A: 0=A00, 1=A01, 2=A10, 3=A11
        # Local B: 0=B00, 1=B01, 2=B10, 3=B11
        # Local C: 0=C00, 1=C01, 2=C10, 3=C11
        ([0, 3], [0, 3], [0, 3]),   # M1
        ([2, 3], [0], [2, 3]),       # M2
        ([0], [1, 3], [1, 3]),       # M3
        ([3], [2, 0], [0, 2]),       # M4
        ([0, 1], [3], [0, 1]),       # M5
        ([2, 0], [0, 1], [3]),       # M6
        ([1, 3], [2, 3], [0]),       # M7
    ]
    
    # With signs for C:
    # M1: +C00, +C11
    # M2: +C10, -C11
    # M3: +C01, +C11
    # M4: +C00, +C10
    # M5: -C00, +C01
    # M6: +C11
    # M7: +C00
    
    c_signs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},              # M6
        {0: 1},              # M7
    ]
    
    # Define blocks: (a_row_start, a_col_start, b_row_start, b_col_start)
    # A: 2 rows, 4 cols -> 2 groups of 2 cols
    # B: 4 rows, 6 cols -> 2 groups of 2 rows, 3 groups of 2 cols
    blocks = []
    for a_col_group in range(2):
        for b_row_group in range(2):
            for b_col_group in range(3):
                if a_col_group == 0 and b_row_group == 1:
                    continue  # A_left only multiplies with B_top
                if a_col_group == 1 and b_row_group == 0:
                    continue  # A_right only multiplies with B_bot
                a_row_start = 0
                a_col_start = a_col_group * 2
                b_row_start = b_row_group * 2
                b_col_start = b_col_group * 2
                blocks.append((a_row_start, a_col_start, b_row_start, b_col_start))
    
    col = 0
    for (a_rs, a_cs, b_rs, b_cs) in blocks:
        # Map local indices to global
        # A local 0->(a_rs, a_cs), 1->(a_rs, a_cs+1), 2->(a_rs+1, a_cs), 3->(a_rs+1, a_cs+1)
        a_map = [
            a_rs * m + a_cs,
            a_rs * m + a_cs + 1,
            (a_rs + 1) * m + a_cs,
            (a_rs + 1) * m + a_cs + 1
        ]
        # B local 0->(b_rs, b_cs), 1->(b_rs, b_cs+1), 2->(b_rs+1, b_cs), 3->(b_rs+1, b_cs+1)
        b_map = [
            b_rs * p + b_cs,
            b_rs * p + b_cs + 1,
            (b_rs + 1) * p + b_cs,
            (b_rs + 1) * p + b_cs + 1
        ]
        # C local 0->(b_cs, a_rs), 1->(b_cs+1, a_rs), 2->(b_cs, a_rs+1), 3->(b_cs+1, a_rs+1)
        # W index: k*n + i
        w_map = [
            b_cs * n + a_rs,
            (b_cs + 1) * n + a_rs,
            b_cs * n + (a_rs + 1),
            (b_cs + 1) * n + (a_rs + 1)
        ]
        
        for term_idx in range(7):
            a_idxs, b_idxs, c_idxs = strassen_terms[term_idx]
            signs = c_signs[term_idx]
            
            for idx in a_idxs:
                U[a_map[idx]][col] += 1
            for idx in b_idxs:
                V[b_map[idx]][col] += 1
            for idx in c_idxs:
                if idx in signs:
                    W[w_map[idx]][col] += signs[idx]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
