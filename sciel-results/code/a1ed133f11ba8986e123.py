"""Matrix multiplication 4,4,9 using recursive Strassen decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base case: 2x2x2 Strassen
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2 multiplication
        # U is 4x7, V is 4x7, W is 4x7
        # Convention: U[i*2+j, k], V[j*2+k, c], W[k*2+i, c]
        # Wait, let me be careful about the index conventions.
        # For n x m x p: U is n*m x rank, V is m*p x rank, W is p*n x rank
        # For 2x2x2: U is 4x7, V is 4x7, W is 4x7
        
        # Strassen's 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each of the 7 factors, define the linear combinations
        # U[row, c] = coefficient for A[row] in factor c
        # V[row, c] = coefficient for B[row] in factor c
        # W[row, c] = coefficient for C[row] in factor c
        
        # A indices: A11=0, A12=1, A21=2, A22=3
        # B indices: B11=0, B12=1, B21=2, B22=3
        # C indices: C11=0, C12=1, C21=2, C22=3
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (A11 + A22)(B11 + B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # C11 and C22 both get +M1
        
        # M2: (A21 + A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1  # C21 gets +M2, C22 gets -M2
        
        # M3: A11(B21 - B22)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1  # C12 gets +M3, C22 gets +M3
        
        # M4: A22(B12 - B11)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1  # C11 gets +M4, C21 gets +M4
        
        # M5: (A11 + A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1  # C11 gets -M5, C12 gets +M5
        
        # M6: (A21 - A11)(B11 + B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # C22 gets +M6
        
        # M7: (A12 - A22)(B21 + B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1  # C11 gets +M7
        
        return U, V, W
    
    # Recursive Strassen for n x m x p where n=m=p is a power of 2
    def strassen_recursive(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        # Split into halves
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        # Get factors for the smaller case
        U_sub, V_sub, W_sub = strassen_recursive(n2, m2, p2)
        rank_sub = len(U_sub[0])
        
        # For the 7 Strassen multiplications, each requires a recursive call
        # with the same dimensions but for the block matrices
        # We need to define how the blocks map to the original indices
        
        # A is n x m, split into 4 blocks of n2 x m2
        # A11: rows 0..n2-1, cols 0..m2-1
        # A12: rows 0..n2-1, cols m2..m-1
        # A21: rows n2..n-1, cols 0..m2-1
        # A22: rows n2..n-1, cols m2..m-1
        
        # B is m x p, split into 4 blocks of m2 x p2
        # B11: rows 0..m2-1, cols 0..p2-1
        # B12: rows 0..m2-1, cols p2..p-1
        # B21: rows m2..m-1, cols 0..p2-1
        # B22: rows m2..m-1, cols p2..p-1
        
        # C is p x n, split into 4 blocks of p2 x n2
        # C11: rows 0..p2-1, cols 0..n2-1
        # C12: rows 0..p2-1, cols n2..n-1
        # C21: rows p2..p-1, cols 0..n2-1
        # C22: rows p2..p-1, cols n2..n-1
        
        # For each of the 7 Strassen products, we define the linear combinations
        # of the A and B blocks, then recursively decompose
        
        U = [[0] * (7 * rank_sub) for _ in range(n * m)]
        V = [[0] * (7 * rank_sub) for _ in range(m * p)]
        W = [[0] * (7 * rank_sub) for _ in range(p * n)]
        
        # Define the 7 block combinations
        # Each combination is a list of (block, coeff) for A, (block, coeff) for B
        # A blocks: 0=A11, 1=A12, 2=A21, 3=A22
        # B blocks: 0=B11, 1=B12, 2=B21, 3=B22
        # C blocks: 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1: (A11+A22)(B11+B22) -> C11 + C22
        # M2: (A21+A22)B11 -> C21 - C22
        # M3: A11(B21-B22) -> C12 + C22
        # M4: A22(B12-B11) -> C11 + C21
        # M5: (A11+A12)B22 -> -C11 + C12
        # M6: (A21-A11)(B11+B12) -> C22
        # M7: (A12-A22)(B21+B22) -> C11
        
        combinations = [
            # (A_coeffs, B_coeffs, C_coeffs)
            # A_coeffs: {block_idx: coeff}
            # B_coeffs: {block_idx: coeff}
            # C_coeffs: {block_idx: coeff}
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),  # M2
            ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),  # M3
            ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),  # M4
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),  # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),  # M6
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),  # M7
        ]
        
        for s in range(7):
            A_coeff = combinations[s][0]
            B_coeff = combinations[s][1]
            C_coeff = combinations[s][2]
            
            # For each sub-factor in the recursive decomposition
            for c in range(rank_sub):
                col = s * rank_sub + c
                
                # Fill U
                for a_block, a_val in A_coeff.items():
                    if a_val != 0:
                        # A_block maps to specific rows in U
                        # A0 (A11): rows i*m2+j for i in 0..n2-1, j in 0..m2-1
                        # A1 (A12): rows i*m2+m2+j for i in 0..n2-1, j in 0..m2-1
                        # A2 (A21): rows (n2+i)*m2+j for i in 0..n2-1, j in 0..m2-1
                        # A3 (A22): rows (n2+i)*m2+m2+j for i in 0..n2-1, j in 0..m2-1
                        for i in range(n2):
                            for j in range(m2):
                                if a_block == 0:
                                    row = i * m2 + j
                                elif a_block == 1:
                                    row = i * m2 + m2 + j
                                elif a_block == 2:
                                    row = (n2 + i) * m2 + j
                                elif a_block == 3:
                                    row = (n2 + i) * m2 + m2 + j
                                else:
                                    continue
                                U[row][col] += a_val * U_sub[i * m2 + j][c]
                
                # Fill V
                for b_block, b_val in B_coeff.items():
                    if b_val != 0:
                        # B_block maps to specific rows in V
                        # B0 (B11): rows j*p2+k for j in 0..m2-1, k in 0..p2-1
                        # B1 (B12): rows j*p2+p2+k for j in 0..m2-1, k in 0..p2-1
                        # B2 (B21): rows (m2+j)*p2+k for j in 0..m2-1, k in 0..p2-1
                        # B3 (B22): rows (m2+j)*p2+p2+k for j in 0..m2-1, k in 0..p2-1
                        for j in range(m2):
                            for k in range(p2):
                                if b_block == 0:
                                    row = j * p2 + k
                                elif b_block == 1:
                                    row = j * p2 + p2 + k
                                elif b_block == 2:
                                    row = (m2 + j) * p2 + k
                                elif b_block == 3:
                                    row = (m2 + j) * p2 + p2 + k
                                else:
                                    continue
                                V[row][col] += b_val * V_sub[j * p2 + k][c]
                
                # Fill W
                for c_block, c_val in C_coeff.items():
                    if c_val != 0:
                        # C_block maps to specific rows in W
                        # C0 (C11): rows k*n2+i for k in 0..p2-1, i in 0..n2-1
                        # C1 (C12): rows k*n2+n2+i for k in 0..p2-1, i in 0..n2-1
                        # C2 (C21): rows (p2+k)*n2+i for k in 0..p2-1, i in 0..n2-1
                        # C3 (C22): rows (p2+k)*n2+n2+i for k in 0..p2-1, i in 0..n2-1
                        for k in range(p2):
                            for i in range(n2):
                                if c_block == 0:
                                    row = k * n2 + i
                                elif c_block == 1:
                                    row = k * n2 + n2 + i
                                elif c_block == 2:
                                    row = (p2 + k) * n2 + i
                                elif c_block == 3:
                                    row = (p2 + k) * n2 + n2 + i
                                else:
                                    continue
                                W[row][col] += c_val * W_sub[k * n2 + i][c]
        
        return U, V, W
    
    # Handle the 4x4x9 case
    # We can view this as 4x4x4 and 4x4x5, or 4x4x3 and 4x4x6, etc.
    # Better: split into 4x4x4 + 4x4x1 + 4x4x1 + 4x4x1 + 4x4x1 + 4x4x1
    # Or: 4x4x9 = 4x4x8 + 4x4x1
    # 4x4x8: can use Strassen-like approach
    # Actually, let's split into blocks of p=2 or p=4
    
    # Let's split 4x4x9 into 4x4x4 + 4x4x4 + 4x4x1
    # For 4x4x4: use strassen_recursive(4,4,4) which gives rank 49
    # For 4x4x1: this is just a 4x4 matrix times a 1x4 vector, or 4x4 times 4x1
    # Wait, 4x4 x 4x1 = 4x1. Rank is at most 4.
    # Actually, for A (4x4) * B (4x1) = C (4x1)
    # C_i = sum_j A_ij B_j
    # This has rank 4 (each C_i depends on all A_ij and B_j)
    # U[i*4+j, k], V[j*1+k, k], W[k*4+i, k] for k=0..3
    # U[i*4+j, j] = 1, V[j, j] = 1, W[j*4+i, j] = 1
    
    def block_4x4x1():
        # A is 4x4, B is 4x1, C is 4x1
        # U is 16x4, V is 4x4, W is 4x4
        U = [[0]*4 for _ in range(16)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(4)]
        for j in range(4):
            for i in range(4):
                U[i*4+j][j] = 1
            V[j][j] = 1
            for i in range(4):
                W[j*4+i][j] = 1
        return U, V, W
    
    # Build 4x4x9 as 4x4x4 + 4x4x4 + 4x4x1
    U_444, V_444, W_444 = strassen_recursive(4, 4, 4)
    rank_444 = len(U_444[0])
    
    U_441, V_441, W_441 = block_44x1()
    rank_441 = len(U_441[0])
    
    total_rank = 2 * rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # First 4x4x4 block: columns 0..rank_444-1
    # This corresponds to B's first 4 columns
    for row in range(n * m):
        for c in range(rank_444):
            U[row][c] = U_444[row][c]
    for row in range(4 * 4):  # m*p2 = 4*4
        for c in range(rank_444):
            V[row][c] = V_444[row][c]
    for row in range(4 * 4):  # p2*n = 4*4
        for c in range(rank_444):
            W[row][c] = W_444[row][c]
    
    # Second 4x4x4 block: columns rank_444..2*rank_444-1
    # This corresponds to B's next 4 columns (columns 4..7)
    offset = rank_444
    for row in range(n * m):
        for c in range(rank_444):
            U[row][offset + c] = U_444[row][c]
    for row in range(4 * 4):  # m*p2 = 4*4
        for c in range(rank_444):
            V[row + 4 * 4][offset + c] = V_444[row][c]  # B rows 4..7
    for row in range(4 * 4):  # p2*n = 4*4
        for c in range(rank_444):
            W[row + 4 * 4][offset + c] = W_444[row][c]  # C rows 4..7
    
    # Third 4x4x1 block: columns 2*rank_444..2*rank_444+rank_441-1
    # This corresponds to B's last column (column 8)
    offset2 = 2 * rank_444
    for row in range(n * m):
        for c in range(rank_441):
            U[row][offset2 + c] = U_441[row][c]
    for row in range(4):  # m*1 = 4
        for c in range(rank_441):
            V[row + 4 * 4 + 4 * 4][offset2 + c] = V_441[row][c]  # B row 8
    for row in range(4):  # 1*n = 4
        for c in range(rank_441):
            W[row + 4 * 4 + 4 * 4][offset2 + c] = W_441[row][c]  # C row 8
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
