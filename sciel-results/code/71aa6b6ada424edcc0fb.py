import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a zero matrix
    def zero_mat(rows, cols):
        return [[0] * cols for _ in range(rows)]
        
    # Decomposition for 2x2x2 Strassen (Rank 7)
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    def get_strassen_2x2x2():
        U = zero_mat(4, 7)
        V = zero_mat(4, 7)
        W = zero_mat(4, 7)
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A11(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = A22(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Decomposition for 2x2x1 Schoolbook (Rank 4)
    # A(2x2), B(2x1) -> C(2x1)
    # U: 4x4, V: 2x4, W: 2x4
    # U[i*2+j, c], V[j*1+k, c] -> V[j, c], W[k*2+i, c] -> W[i, c]
    # Wait, W index is k*n+i. n=2. k=0. W[i, c].
    def get_schoolbook_2x2x1():
        U = zero_mat(4, 4)
        V = zero_mat(2, 4)
        W = zero_mat(2, 4)
        c = 0
        for i in range(2):
            for j in range(2):
                U[i*2+j][c] = 1
                V[j][c] = 1
                W[i][c] = 1
                c += 1
        return U, V, W

    # Decomposition for 2x1x3 Schoolbook (Rank 1)
    # A(2x1), B(1x3) -> C(2x3)
    # U: 2x1, V: 3x1, W: 6x1
    # U[i*1+j, c] -> U[i, c]
    # V[j*3+k, c] -> V[k, c] (j=0)
    # W[k*2+i, c]
    def get_schoolbook_2x1x3():
        U = zero_mat(2, 1)
        V = zero_mat(3, 1)
        W = zero_mat(6, 1)
        for i in range(2):
            U[i][0] = 1
            for k in range(3):
                V[k][0] = 1
                W[k*2+i][0] = 1
        return U, V, W

    # Decomposition for 1x2x3 Schoolbook (Rank 2)
    # A(1x2), B(2x3) -> C(1x3)
    # U: 2x2, V: 6x2, W: 3x2
    # U[0*2+j, c] -> U[j, c]
    # V[j*3+k, c]
    # W[k*1+0, c] -> W[k, c]
    def get_schoolbook_1x2x3():
        U = zero_mat(2, 2)
        V = zero_mat(6, 2)
        W = zero_mat(3, 2)
        for j in range(2):
            U[j][j] = 1
            for k in range(3):
                V[j*3+k][j] = 1
                W[k][j] = 1
        return U, V, W

    # Decomposition for 1x3x3 Schoolbook (Rank 3)
    # A(1x3), B(3x3) -> C(1x3)
    # U: 3x3, V: 9x3, W: 3x3
    def get_schoolbook_1x3x3():
        U = zero_mat(3, 3)
        V = zero_mat(9, 3)
        W = zero_mat(3, 3)
        for j in range(3):
            U[j][j] = 1
            for k in range(3):
                V[j*3+k][j] = 1
                W[k][j] = 1
        return U, V, W

    # Decomposition for 2x2x3 (Rank 11)
    # Split B(2x3) into B_L(2x2) and B_R(2x1)
    # C_L(2x2) = A(2x2) B_L(2x2) -> Rank 7
    # C_R(2x1) = A(2x2) B_R(2x1) -> Rank 4
    def get_decomp_2x2x3():
        U1, V1, W1 = get_strassen_2x2x2()
        U2, V2, W2 = get_schoolbook_2x2x1()
        
        U = zero_mat(4, 11)
        V = zero_mat(6, 11) # B is 2x3 -> 6 entries
        W = zero_mat(6, 11) # C is 2x3 -> 6 entries
        
        # Copy U1, V1, W1 (cols 0-6)
        # V1 is 4x7 (2x2). B_L is top-left 2x2 of B.
        # B indices: j*3+k.
        # B_L: j in 0,1; k in 0,1. Indices: 0,1, 3,4.
        # V1 indices: j*2+k.
        # Map V1[j*2+k] to V[j*3+k].
        for j in range(2):
            for k in range(2):
                V[j*3+k][:7] = V1[j*2+k][:7]
                
        # W1 is 4x7 (2x2). C_L is top-left 2x2 of C.
        # C indices: k*2+i.
        # C_L: k in 0,1; i in 0,1. Indices: 0,1, 2,3.
        # W1 indices: k*2+i.
        # Map W1[k*2+i] to W[k*2+i].
        for k in range(2):
            for i in range(2):
                W[k*2+i][:7] = W1[k*2+i][:7]
                
        # Copy U2, V2, W2 (cols 7-10)
        # U2 is 4x4.
        for r in range(4):
            U[:, 7+r] = U2[:, r]
            
        # V2 is 2x4 (2x1). B_R is right 2x1 of B.
        # B_R: j in 0,1; k=2. Indices: 2, 5.
        # V2 indices: j.
        # Map V2[j] to V[j*3+2].
        for j in range(2):
            V[j*3+2][7:11] = V2[j][7:11] # Wait, V2 is 2x4. V2[j] is row j.
            # Correction: V2[j][r]
            for r in range(4):
                V[j*3+2][7+r] = V2[j][r]
                
        # W2 is 2x4 (2x1). C_R is right 2x1 of C.
        # C_R: k in 0,1; i in 0,1? No, C is 2x3.
        # C_R is cols 2? No, C is 2x3.
        # C_L is 2x2 (cols 0,1). C_R is 2x1 (col 2).
        # C indices: k*2+i.
        # C_R: k in 0,1; i=2? No, i is row index of C.
        # C is 2x3. Rows i=0,1. Cols k=0,1,2.
        # C_L: k=0,1. C_R: k=2.
        # W indices: k*n+i = k*2+i.
        # C_R indices: k=2. i=0,1. Indices: 2*2+0=4, 2*2+1=5.
        # W2 indices: k*2+i? No, W2 is for 2x2x1.
        # 2x2x1: A(2x2), B(2x1) -> C(2x1).
        # C rows i=0,1. Col k=0.
        # W2 indices: k*2+i. k=0.
        # Map W2[i] to W[2*2+i] = W[4+i].
        for i in range(2):
            for r in range(4):
                W[4+i][7+r] = W2[i][r]
                
        return U, V, W

    # Decomposition for 2x3x3 (Rank 12)
    # Split A(2x3) into A_L(2x1) and A_R(2x2)
    # Split B(3x3) into B_L(1x3) and B_R(2x3)
    # C_L(2x3) = A_L(2x1) B_L(1x3) -> Rank 1
    # C_R(2x3) = A_R(2x2) B_R(2x3) -> Rank 11
    def get_decomp_2x3x3():
        U1, V1, W1 = get_schoolbook_2x1x3() # Rank 1
        U2, V2, W2 = get_decomp_2x2x3()     # Rank 11
        
        U = zero_mat(6, 12) # A is 2x3 -> 6 entries
        V = zero_mat(9, 12) # B is 3x3 -> 9 entries
        W = zero_mat(6, 12) # C is 2x3 -> 6 entries
        
        # Copy U1, V1, W1 (cols 0)
        # U1 is 2x1. A_L is left 2x1 of A.
        # A indices: i*3+j.
        # A_L: i in 0,1; j=0. Indices: 0, 3.
        # U1 indices: i.
        # Map U1[i] to U[i*3].
        for i in range(2):
            U[i*3][0] = U1[i][0]
            
        # V1 is 3x1. B_L is top 1x3 of B.
        # B indices: j*3+k.
        # B_L: j=0; k in 0,1,2. Indices: 0,1,2.
        # V1 indices: k.
        # Map V1[k] to V[k].
        for k in range(3):
            V[k][0] = V1[k][0]
            
        # W1 is 6x1. C_L is left 2x3 of C? No, C is 2x3.
        # C_L is the whole C?
        # A_L(2x1) B_L(1x3) -> C(2x3).
        # Yes, C is 2x3.
        # W1 indices: k*2+i.
        # Map W1[k*2+i] to W[k*2+i].
        for k in range(3):
            for i in range(2):
                W[k*2+i][0] = W1[k*2+i][0]
                
        # Copy U2, V2, W2 (cols 1-11)
        # U2 is 4x11. A_R is right 2x2 of A.
        # A_R: i in 0,1; j in 1,2. Indices: 1,2, 4,5.
        # U2 indices: i*2+j.
        # Map U2[i*2+j] to U[i*3+j+1].
        for i in range(2):
            for j in range(2):
                for r in range(11):
                    U[i*3+j+1][1+r] = U2[i*2+j][r]
                    
        # V2 is 6x11. B_R is bottom 2x3 of B.
        # B_R: j in 1,2; k in 0,1,2. Indices: 3,4,5, 6,7,8.
        # V2 indices: j*3+k. (j in 0,1).
        # Map V2[j*3+k] to V[(j+1)*3+k].
        for j in range(2):
            for k in range(3):
                for r in range(11):
                    V[(j+1)*3+k][1+r] = V2[j*3+k][r]
                    
        # W2 is 6x11. C_R is the whole C?
        # A_R(2x2) B_R(2x3) -> C(2x3).
        # Yes.
        # W2 indices: k*2+i.
        # Map W2[k*2+i] to W[k*2+i].
        for k in range(3):
            for i in range(2):
                for r in range(11):
                    W[k*2+i][1+r] = W2[k*2+i][r]
                    
        return U, V, W

    # Main construction
    # n=3, m=5, p=6
    # Split n=3 into 2+1
    # Split m=5 into 2+3
    # Split p=6 into 3+3
    
    # A: 3x5
    # A11 (2x2), A12 (2x3)
    # A21 (1x2), A22 (1x3)
    
    # B: 5x6
    # B11 (2x3), B12 (2x3)
    # B21 (3x3), B22 (3x3)
    
    # C: 3x6
    # C11 (2x3), C12 (2x3)
    # C21 (1x3), C22 (1x3)
    
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Ranks:
    # C11: 11 (2x2x3) + 12 (2x3x3) = 23
    # C12: 11 + 12 = 23
    # C21: 2 (1x2x3) + 3 (1x3x3) = 5
    # C22: 2 + 3 = 5
    # Total Rank = 56
    
    total_rank = 56
    U = zero_mat(15, total_rank)
    V = zero_mat(30, total_rank)
    W = zero_mat(18, total_rank)
    
    col_offset = 0
    
    # C11
    U1, V1, W1 = get_decomp_2x2x3() # A11 B11
    U2, V2, W2 = get_decomp_2x3x3() # A12 B21
    
    # Map A11 (2x2) to A (3x5)
    # A11: i in 0,1; j in 0,1. Indices: 0,1, 5,6.
    # U1 indices: i*2+j.
    for i in range(2):
        for j in range(2):
            for r in range(11):
                U[i*5+j][col_offset+r] = U1[i*2+j][r]
                
    # Map B11 (2x3) to B (5x6)
    # B11: j in 0,1; k in 0,1,2. Indices: 0,1,2, 6,7,8.
    # V1 indices: j*3+k.
    for j in range(2):
        for k in range(3):
            for r in range(11):
                V[j*6+k][col_offset+r] = V1[j*3+k][r]
                
    # Map C11 (2x3) to C (3x6)
    # C11: k in 0,1,2; i in 0,1. Indices: 0,1, 6,7, 12,13.
    # W1 indices: k*2+i.
    for k in range(3):
        for i in range(2):
            for r in range(11):
                W[k*3+i][col_offset+r] = W1[k*2+i][r]
                
    col_offset += 11
    
    # A12 (2x3) to A (3x5)
    # A12: i in 0,1; j in 2,3,4. Indices: 2,3,4, 7,8,9.
    # U2 indices: i*3+j.
    for i in range(2):
        for j in range(3):
            for r in range(12):
                U[i*5+j+2][col_offset+r] = U2[i*3+j][r]
                
    # B21 (3x3) to B (5x6)
    # B21: j in 2,3,4; k in 0,1,2. Indices: 12,13,14, 18,19,20, 24,25,26.
    # V2 indices: j*3+k.
    for j in range(3):
        for k in range(3):
            for r in range(12):
                V[(j+2)*6+k][col_offset+r] = V2[j*3+k][r]
                
    # C11 (2x3) to C (3x6) - Wait, C11 is the same target.
    # We are adding to the same W block.
    for k in range(3):
        for i in range(2):
            for r in range(12):
                W[k*3+i][col_offset+r] = W2[k*2+i][r]
                
    col_offset += 12
    
    # C12
    U1, V1, W1 = get_decomp_2x2x3() # A11 B12
    U2, V2, W2 = get_decomp_2x3x3() # A12 B22
    
    # A11
    for i in range(2):
        for j in range(2):
            for r in range(11):
                U[i*5+j][col_offset+r] = U1[i*2+j][r]
                
    # B12 (2x3) to B (5x6)
    # B12: j in 0,1; k in 3,4,5. Indices: 3,4,5, 9,10,11.
    for j in range(2):
        for k in range(3):
            for r in range(11):
                V[j*6+k+3][col_offset+r] = V1[j*3+k][r]
                
    # C12 (2x3) to C (3x6)
    # C12: k in 3,4,5; i in 0,1. Indices: 3,4, 9,10, 15,16.
    for k in range(3):
        for i in range(2):
            for r in range(11):
                W[(k+3)*3+i][col_offset+r] = W1[k*2+i][r]
                
    col_offset += 11
    
    # A12
    for i in range(2):
        for j in range(3):
            for r in range(12):
                U[i*5+j+2][col_offset+r] = U2[i*3+j][r]
                
    # B22 (3x3) to B (5x6)
    # B22: j in 2,3,4; k in 3,4,5. Indices: 15,16,17, 21,22,23, 27,28,29.
    for j in range(3):
        for k in range(3):
            for r in range(12):
                V[(j+2)*6+k+3][col_offset+r] = V2[j*3+k][r]
                
    # C12
    for k in range(3):
        for i in range(2):
            for r in range(12):
                W[(k+3)*3+i][col_offset+r] = W2[k*2+i][r]
                
    col_offset += 12
    
    # C21
    U1, V1, W1 = get_schoolbook_1x2x3() # A21 B11
    U2, V2, W2 = get_schoolbook_1x3x3() # A22 B21
    
    # A21 (1x2) to A (3x5)
    # A21: i=2; j in 0,1. Indices: 10, 11.
    # U1 indices: j.
    for j in range(2):
        for r in range(2):
            U[2*5+j][col_offset+r] = U1[j][r]
            
    # B11
    for j in range(2):
        for k in range(3):
            for r in range(2):
                V[j*6+k][col_offset+r] = V1[j*3+k][r]
                
    # C21 (1x3) to C (3x6)
    # C21: k in 0,1,2; i=2. Indices: 2, 8, 14.
    # W1 indices: k.
    for k in range(3):
        for r in range(2):
            W[k*3+2][col_offset+r] = W1[k][r]
            
    col_offset += 2
    
    # A22 (1x3) to A (3x5)
    # A22: i=2; j in 2,3,4. Indices: 12, 13, 14.
    # U2 indices: j.
    for j in range(3):
        for r in range(3):
            U[2*5+j+2][col_offset+r] = U2[j][r]
            
    # B21
    for j in range(3):
        for k in range(3):
            for r in range(3):
                V[(j+2)*6+k][col_offset+r] = V2[j*3+k][r]
                
    # C21
    for k in range(3):
        for r in range(3):
            W[k*3+2][col_offset+r] = W2[k][r]
            
    col_offset += 3
    
    # C22
    U1, V1, W1 = get_schoolbook_1x2x3() # A21 B12
    U2, V2, W2 = get_schoolbook_1x3x3() # A22 B22
    
    # A21
    for j in range(2):
        for r in range(2):
            U[2*5+j][col_offset+r] = U1[j][r]
            
    # B12
    for j in range(2):
        for k in range(3):
            for r in range(2):
                V[j*6+k+3][col_offset+r] = V1[j*3+k][r]
                
    # C22 (1x3) to C (3x6)
    # C22: k in 3,4,5; i=2. Indices: 5, 11, 17.
    for k in range(3):
        for r in range(2):
            W[(k+3)*3+2][col_offset+r] = W1[k][r]
            
    col_offset += 2
    
    # A22
    for j in range(3):
        for r in range(3):
            U[2*5+j+2][col_offset+r] = U2[j][r]
            
    # B22
    for j in range(3):
        for k in range(3):
            for r in range(3):
                V[(j+2)*6+k+3][col_offset+r] = V2[j*3+k][r]
                
    # C22
    for k in range(3):
        for r in range(3):
            W[(k+3)*3+2][col_offset+r] = W2[k][r]
            
    col_offset += 3
    
    return {"U": U, "V": V, "W": W}

