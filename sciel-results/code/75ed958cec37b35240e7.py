"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-78 exact tensor decomposition for 3x4 by 4x7 matrix multiplication.
    
    Strategy:
    - Split the 3x4 matrix into two 3x2 blocks: A = [A1 | A2]
    - Split the 4x7 matrix into two 2x7 blocks: B = [B1; B2]
    - C = A1*B1 + A2*B2
    
    For each 3x2 by 2x7 multiplication:
    - Split 3x2 into 2x2 (top) + 1x2 (bottom)
    - Use Strassen (rank 7) for the 2x2 by 2x7 part (3 Strassen blocks for cols 1-6, 1 schoolbook for col 7)
    - Use schoolbook (rank 4) for the 1x2 by 2x7 part (14 terms total for 1x2x7)
    - Total per 3x2x7 block: 3*7 + 4 + 14 = 39
    - Total for 3x4x7: 2 * 39 = 78
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We'll build U, V, W as lists of lists
    # U: (n*m, rank) = (12, 78)
    # V: (m*p, rank) = (28, 78)  
    # W: (p*n, rank) = (21, 78)
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process two 3x2 by 2x7 blocks
    for block_idx in range(2):
        # Each block handles columns 2*block_idx and 2*block_idx+1 of A, 
        # and rows 2*block_idx and 2*block_idx+1 of B
        
        # --- Part 1: 2x2 by 2x7 using Strassen for 3 blocks of 2x2x2 + schoolbook for 2x2x1 ---
        # Strassen for 2x2x2:
        # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-c),
        # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        
        # For 2x2 by 2x7, we apply Strassen to each pair of output columns
        # Cols 0-1: Strassen block 1
        # Cols 2-3: Strassen block 2
        # Cols 4-5: Strassen block 3
        # Col 6: schoolbook (4 terms)
        
        for strassen_block in range(3):
            # Output columns in the 2x7 result: 2*strassen_block, 2*strassen_block+1
            out_col_offset = 2 * strassen_block
            
            # Strassen M1: (a+d)(e+h)
            # U: rows for A1, A2 (2x2) -> indices in 3x2: row 0,1 of this block
            # For block_idx, A rows are 0,1,2. The 2x2 part uses A rows 0,1.
            # U index for A[i,j]: i*m + j = i*4 + (2*block_idx + j)
            # For 2x2 part: i=0,1; j=0,1 within the 2x2
            U[0*4 + 2*block_idx + 0][col] = 1  # a
            U[0*4 + 2*block_idx + 1][col] = 1  # b? No, wait.
            # Let me re-index. For the 2x2 submatrix of A (rows 0,1 of the 3-row block):
            # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
            # In global U: row = i*4 + col_in_A
            # For block_idx, the 2 columns of A are at positions 2*block_idx and 2*block_idx+1
            # So a = A[0, 2*block_idx], b = A[0, 2*block_idx+1], c = A[1, 2*block_idx], d = A[1, 2*block_idx+1]
            # U indices: 0*4 + 2*block_idx (for a), 0*4 + 2*block_idx+1 (for b), etc.
            
            # Reset and redo properly
            # M1 = (a+d)(e+h)
            U[0*4 + 2*block_idx][col] = 1   # a
            U[1*4 + 2*block_idx+1][col] = 1 # d
            # V: rows for B (2x7). B rows are 2*block_idx, 2*block_idx+1
            # e = B[2*block_idx, 2*strassen_block], h = B[2*block_idx+1, 2*strassen_block+1]
            # V index: j*p + k = j*7 + k
            V[2*block_idx*7 + 2*strassen_block][col] = 1       # e
            V[(2*block_idx+1)*7 + 2*strassen_block+1][col] = 1 # h
            # W: output C (3x7). For this 2x2 block, output rows 0,1
            # C11 = M1+M4-M5+M7, so M1 contributes +1 to C[0, 2*strassen_block]
            # W index: k*n + i = k*3 + i
            W[2*strassen_block*3 + 0][col] = 1  # C[0, 2*strassen_block]
            col += 1
            
            # M2 = (c+d)e
            U[1*4 + 2*block_idx][col] = 1   # c
            U[1*4 + 2*block_idx+1][col] = 1 # d
            V[2*block_idx*7 + 2*strassen_block][col] = 1       # e
            # C21 = M2+M4, so M2 contributes +1 to C[1, 2*strassen_block]
            W[2*strassen_block*3 + 1][col] = 1
            col += 1
            
            # M3 = a(f-h)
            U[0*4 + 2*block_idx][col] = 1   # a
            V[2*block_idx*7 + 2*strassen_block+1][col] = 1     # f
            V[(2*block_idx+1)*7 + 2*strassen_block+1][col] = -1 # -h
            # C12 = M3+M5, so M3 contributes +1 to C[0, 2*strassen_block+1]
            W[(2*strassen_block+1)*3 + 0][col] = 1
            col += 1
            
            # M4 = d(g-c)
            U[1*4 + 2*block_idx+1][col] = 1 # d
            V[2*block_idx*7 + 2*strassen_block+1][col] = 1     # g
            V[(2*block_idx+1)*7 + 2*strassen_block][col] = -1  # -c
            # C11 = M1+M4-M5+M7, C21 = M2+M4, so M4 contributes +1 to both
            W[2*strassen_block*3 + 0][col] = 1
            W[2*strassen_block*3 + 1][col] = 1
            col += 1
            
            # M5 = (a+b)h
            U[0*4 + 2*block_idx][col] = 1   # a
            U[0*4 + 2*block_idx+1][col] = 1 # b
            V[(2*block_idx+1)*7 + 2*strassen_block+1][col] = 1 # h
            # C11 = M1+M4-M5+M7 (M5 contributes -1), C12 = M3+M5 (M5 contributes +1)
            W[2*strassen_block*3 + 0][col] = -1
            W[(2*strassen_block+1)*3 + 0][col] = 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[1*4 + 2*block_idx][col] = 1   # c
            U[0*4 + 2*block_idx][col] = -1  # -a
            V[2*block_idx*7 + 2*strassen_block][col] = 1     # e
            V[2*block_idx*7 + 2*strassen_block+1][col] = 1   # f
            # C22 = M1-M2+M3+M6, so M6 contributes +1 to C[1, 2*strassen_block+1]
            W[(2*strassen_block+1)*3 + 1][col] = 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[0*4 + 2*block_idx+1][col] = 1 # b
            U[1*4 + 2*block_idx+1][col] = -1 # -d
            V[2*block_idx*7 + 2*strassen_block+1][col] = 1   # g
            V[(2*block_idx+1)*7 + 2*strassen_block+1][col] = 1 # h
            # C11 = M1+M4-M5+M7 (M7 contributes +1), C22 = M1-M2+M3+M6 (M7 does NOT contribute to C22)
            # Wait, let me recheck Strassen:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # So M7 only contributes to C11.
            W[2*strassen_block*3 + 0][col] = 1
            col += 1
        
        # Schoolbook for 2x2x1 (column 6 of the 2x7 result)
        # 4 terms: a*e, b*f, c*g, d*h where e,f,g,h are B[*,6]
        # a = A[0, 2*block_idx], b = A[0, 2*block_idx+1], c = A[1, 2*block_idx], d = A[1, 2*block_idx+1]
        # e = B[2*block_idx, 6], f = B[2*block_idx+1, 6], g = B[2*block_idx, 6]... 
        # Wait, B is 2x7 for this block. B rows are 2*block_idx and 2*block_idx+1.
        # B[2*block_idx, 6] and B[2*block_idx+1, 6]
        # C[0,6] = a*B[2*block_idx,6] + b*B[2*block_idx+1,6]
        # C[1,6] = c*B[2*block_idx,6] + d*B[2*block_idx+1,6]
        
        # Term 1: a * B[2*block_idx, 6]
        U[0*4 + 2*block_idx][col] = 1
        V[2*block_idx*7 + 6][col] = 1
        W[6*3 + 0][col] = 1  # C[0,6]
        col += 1
        
        # Term 2: b * B[2*block_idx+1, 6]
        U[0*4 + 2*block_idx+1][col] = 1
        V[(2*block_idx+1)*7 + 6][col] = 1
        W[6*3 + 0][col] = 1  # C[0,6]
        col += 1
        
        # Term 3: c * B[2*block_idx, 6]
        U[1*4 + 2*block_idx][col] = 1
        V[2*block_idx*7 + 6][col] = 1
        W[6*3 + 1][col] = 1  # C[1,6]
        col += 1
        
        # Term 4: d * B[2*block_idx+1, 6]
        U[1*4 + 2*block_idx+1][col] = 1
        V[(2*block_idx+1)*7 + 6][col] = 1
        W[6*3 + 1][col] = 1  # C[1,6]
        col += 1
        
        # --- Part 2: 1x2 by 2x7 using schoolbook (14 terms) ---
        # A row 2 (the 1x2 part): A[2, 2*block_idx], A[2, 2*block_idx+1]
        # B rows 2*block_idx, 2*block_idx+1 (2x7)
        # C[2, k] = A[2, 2*block_idx]*B[2*block_idx, k] + A[2, 2*block_idx+1]*B[2*block_idx+1, k]
        
        for k in range(7):
            # Term: A[2, 2*block_idx] * B[2*block_idx, k]
            U[2*4 + 2*block_idx][col] = 1
            V[2*block_idx*7 + k][col] = 1
            W[k*3 + 2][col] = 1  # C[2, k]
            col += 1
            
            # Term: A[2, 2*block_idx+1] * B[2*block_idx+1, k]
            U[2*4 + 2*block_idx+1][col] = 1
            V[(2*block_idx+1)*7 + k][col] = 1
            W[k*3 + 2][col] = 1  # C[2, k]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
