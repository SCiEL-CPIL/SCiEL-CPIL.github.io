"""Program for matrix multiplication 3,4,4 with rank 44 decomposition."""

def construct():
    n, m, p = 3, 4, 4
    # Shapes:
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank)
    # W index convention: W[k*n + i, r] where k is col (0..3), i is row (0..2)
    
    # We will build the matrices as lists of lists
    # First, determine total rank
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
        nonlocal col
        U[u_idx][col] = coeff_u
        V[v_idx][col] = coeff_v
        W[w_idx][col] = coeff_w
        col += 1

    # Indices:
    # A[i, j] -> U[i*m + j]
    # B[j, k] -> V[j*p + k]
    # C[i, k] -> W[k*n + i]
    
    # --- Part 1: Top 2x4 multiplication ---
    # C[0:2, 0:4] = A[0:2, 0:4] B
    # Split into 4 2x2 multiplications using Strassen
    
    # We need a function to add a Strassen multiplication
    # a_indices: list of 4 U indices for the 2x2 matrix A_block
    # b_indices: list of 4 V indices for the 2x2 matrix B_block
    # c_indices: list of 4 W indices for the 2x2 matrix C_block
    # The C_block will be added to the existing W entries (summed)
    
    def add_strassen(a_idx, b_idx, c_idx):
        # a_idx[0]=00, [1]=01, [2]=10, [3]=11
        # b_idx[0]=00, [1]=01, [2]=10, [3]=11
        # c_idx[0]=00, [1]=01, [2]=10, [3]=11
        
        # M1 = (a00 + a11) * (b00 + b11)
        # This requires 2 terms in U/V? No, Strassen is a decomposition of the tensor.
        # The term (a00+a11)*(b00+b11) is bilinear.
        # (a00 + a11) is a linear combination of U entries.
        # (b00 + b11) is a linear combination of V entries.
        # The product contributes to W entries.
        
        # We can't just "add" linear combinations directly in the U,V,W matrices 
        # unless we expand them into rank-1 terms or pre-compute the linear combos.
        # The standard tensor decomposition assigns 1 to specific U and V entries.
        # Strassen's algorithm defines 7 multiplications of linear forms.
        # L1 = a00 + a11, L2 = b00 + b11. Product L1*L2.
        # In the tensor decomposition, we need to represent the bilinear map.
        # The entry in U for this column will be the coefficients of the linear form for A.
        # The entry in V will be the coefficients for B.
        # The entry in W will be the coefficients for C.
        
        # For M1:
        # U part: a00 and a11 have coeff 1.
        # V part: b00 and b11 have coeff 1.
        # W part: c00 (+1), c01 (+1), c10 (+1), c11 (-1) ? 
        # Wait, Z = M1 + M4 - M5 + M7 etc.
        # So for column corresponding to M1:
        # U[a00] += 1, U[a11] += 1
        # V[b00] += 1, V[b11] += 1
        # W[c00] += 1, W[c01] += 1, W[c10] += 1, W[c11] += -1 ? 
        # No, the output of Strassen is Z.
        # Z00 gets M1. Z01 gets M3+M5.
        # So for the term M1, it contributes +1 to Z00, -1 to Z11? 
        # Let's check Z formulas again.
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # So for the column representing M1:
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: c00=1, c11=1  (since M1 adds to Z00 and Z11)
        
        # For M2: (a10+a11)*b00
        # U: a10=1, a11=1
        # V: b00=1
        # W: c10=1, c11=-1 (M2 adds to Z10, subtracts from Z11)
        
        # For M3: a00*(b01-b11)
        # U: a00=1
        # V: b01=1, b11=-1
        # W: c01=1, c11=1
        
        # For M4: a11*(b10-b00)
        # U: a11=1
        # V: b10=1, b00=-1
        # W: c00=1, c10=1
        
        # For M5: (a00+a01)*b11
        # U: a00=1, a01=1
        # V: b11=1
        # W: c00=-1, c01=1
        
        # For M6: (a10-a00)*(b00+b01)
        # U: a10=1, a00=-1
        # V: b00=1, b01=1
        # W: c11=1
        
        # For M7: (a01-a11)*(b10+b11)
        # U: a01=1, a11=-1
        # V: b10=1, b11=1
        # W: c00=1, c11=-1

        # Now we map this to the global U, V, W arrays.
        # We need to add these coefficients to the current column.
        
        # M1
        U[a_idx[0]][col] += 1; U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1; V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1; W[c_idx[3]][col] += 1
        col += 1
        
        # M2
        U[a_idx[2]][col] += 1; U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1; W[c_idx[3]][col] += -1
        col += 1
        
        # M3
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1; V[b_idx[3]][col] += -1
        W[c_idx[1]][col] += 1; W[c_idx[3]][col] += 1
        col += 1
        
        # M4
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1; V[b_idx[0]][col] += -1
        W[c_idx[0]][col] += 1; W[c_idx[2]][col] += 1
        col += 1
        
        # M5
        U[a_idx[0]][col] += 1; U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += -1; W[c_idx[1]][col] += 1
        col += 1
        
        # M6
        U[a_idx[2]][col] += 1; U[a_idx[0]][col] += -1
        V[b_idx[0]][col] += 1; V[b_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M7
        U[a_idx[1]][col] += 1; U[a_idx[3]][col] += -1
        V[b_idx[2]][col] += 1; V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1; W[c_idx[3]][col] += -1
        col += 1

    # Define indices for the 4 2x2 blocks
    # A_top is rows 0,1. Cols 0,1,2,3.
    # Block 1 (TL of result): A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0,1 (row 0), 4,5 (row 1) -> [0, 1, 4, 5]
    # B indices: 0,1 (col 0, row 0,1? No B is 4x4. j=0..3, k=0..3)
    # B[j,k] index is j*4+k.
    # B[0:2, 0:2] means j in 0,1; k in 0,1.
    # Indices: 0, 1 (j=0), 4, 5 (j=1) -> [0, 1, 4, 5]
    # C indices: C[i,k] index is k*3+i.
    # C[0:2, 0:2] means i in 0,1; k in 0,1.
    # k=0: i=0->0, i=1->1. k=1: i=0->3, i=1->4.
    # Indices: 0, 1, 3, 4.
    
    # Wait, the block structure of B for multiplication with A_top (2x4):
    # A_top = [A11 A12] (2x2 blocks)
    # B = [[B11 B12]
    #      [B21 B22]] (2x2 blocks)
    # B11 is B[0:2, 0:2]. B12 is B[0:2, 2:4]. B21 is B[2:4, 0:2]. B22 is B[2:4, 2:4].
    
    # Sub-problem 1: A11 * B11 -> contributes to C[0:2, 0:2]
    a11_idx = [0, 1, 4, 5] # A[0,0], A[0,1], A[1,0], A[1,1]
    b11_idx = [0, 1, 4, 5] # B[0,0], B[0,1], B[1,0], B[1,1]
    c11_idx = [0, 3, 1, 4] # C[0,0], C[0,1], C[1,0], C[1,1] 
                           # Order: 00, 01, 10, 11
                           # C[0,0] is k=0,i=0 -> 0
                           # C[0,1] is k=1,i=0 -> 3
                           # C[1,0] is k=0,i=1 -> 1
                           # C[1,1] is k=1,i=1 -> 4
                           # So c_idx = [0, 3, 1, 4]
    
    add_strassen(a11_idx, b11_idx, c11_idx)
    
    # Sub-problem 2: A12 * B21 -> contributes to C[0:2, 0:2]
    # A12 is A[0:2, 2:4]. Indices: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7.
    a12_idx = [2, 3, 6, 7]
    # B21 is B[2:4, 0:2]. Indices: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13.
    b21_idx = [8, 9, 12, 13]
    # C[0:2, 0:2] is same c11_idx.
    add_strassen(a12_idx, b21_idx, c11_idx)
    
    # Sub-problem 3: A11 * B12 -> contributes to C[0:2, 2:4]
    # B12 is B[0:2, 2:4]. Indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7.
    b12_idx = [2, 3, 6, 7]
    # C[0:2, 2:4] means k in 2,3.
    # k=2: i=0->6, i=1->7. k=3: i=0->9, i=1->10.
    # C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10.
    c12_idx = [6, 9, 7, 10]
    add_strassen(a11_idx, b12_idx, c12_idx)
    
    # Sub-problem 4: A12 * B22 -> contributes to C[0:2, 2:4]
    # B22 is B[2:4, 2:4]. Indices: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15.
    b22_idx = [10, 11, 14, 15]
    add_strassen(a12_idx, b22_idx, c12_idx)
    
    # --- Part 2: Bottom 1x4 multiplication ---
    # C[2, 0:4] = A[2, 0:4] B
    # A[2, j] index is 2*4 + j = 8+j.
    # B[j, k] index is j*4 + k.
    # C[2, k] index is k*3 + 2.
    
    for k in range(4): # Column of C
        w_idx = k * 3 + 2
        for j in range(4): # Row of B / Col of A
            u_idx = 8 + j
            v_idx = j * 4 + k
            add_term(u_idx, v_idx, w_idx)
            
    return {"U": U, "V": V, "W": W}
