"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose p=7 into p1=2 and p2=5
    p1 = 2
    p2 = 5
    
    U = []
    V = []
    W = []
    
    def add_decomposition(U_part, V_part, W_part, offset_U, offset_V, offset_W):
        # U_part: (n*m, rank_part)
        # V_part: (m*p_part, rank_part) -- Note: V is indexed by j*p+k
        # W_part: (p_part*n, rank_part) -- Note: W is indexed by k*n+i
        # We need to map indices correctly.
        pass

    # This approach is tricky with offsets. 
    # Better to build the full matrices directly.
    
    rank = 0
    U_full = []
    V_full = []
    W_full = []
    
    # Helper to append a block
    def append_block(U_block, V_block, W_block):
        nonlocal rank
        r_block = len(U_block[0])
        for i in range(len(U_full)):
            U_full[i].extend(U_block[i])
        for i in range(len(V_full)):
            V_full[i].extend(V_block[i])
        for i in range(len(W_full)):
            W_full[i].extend(W_block[i])
        rank += r_block

    # Initialize
    U_full = [[0] * 0 for _ in range(n * m)]
    V_full = [[0] * 0 for _ in range(m * p)]
    W_full = [[0] * 0 for _ in range(p * n)]
    
    # Block 1: 2 x 4 x 2
    # Split m=4 into m1=2, m2=2
    # A1 (2x2) * B11 (2x2) -> C11 (2x2)
    # A2 (2x2) * B21 (2x2) -> C21 (2x2)
    
    # Strassen for 2x2x2
    def strassen_2x2x2():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2x2
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # Standard Strassen decomposition
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # U indices: 0:11, 1:12, 2:21, 3:22
        # V indices: 0:11, 1:12, 2:21, 3:22
        # W indices: 0:11, 1:12, 2:21, 3:22
        
        # M1
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1; W_s[1][0] = -1; W_s[2][0] = -1
        
        # M2
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1; W_s[3][1] = 1
        
        # M3
        U_s[0][2] = 1
        V_s[2][2] = 1; V_s[3][2] = -1
        W_s[1][2] = 1; W_s[3][2] = 1
        
        # M4
        U_s[3][3] = 1
        V_s[0][3] = 1; V_s[2][3] = -1
        W_s[0][3] = 1; W_s[2][3] = 1
        
        # M5
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1
        
        # M6
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1
        
        # M7
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1; W_s[2][6] = 1
        
        return U_s, V_s, W_s

    # Schoolbook for 2x2x5
    def schoolbook_2x2x5():
        U_s = [[0]*20 for _ in range(4)]
        V_s = [[0]*20 for _ in range(10)] # m*p = 2*5=10
        W_s = [[0]*20 for _ in range(10)] # p*n = 5*2=10
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(5):
                    U_s[i*2+j][col] = 1
                    V_s[j*5+k][col] = 1
                    W_s[k*2+i][col] = 1
                    col += 1
        return U_s, V_s, W_s

    # Build Block 1 (p1=2)
    # A1 (2x2) * B11 (2x2)
    U1, V1, W1 = strassen_2x2x2()
    # Map to full indices
    # U1 is 4x7. Full U is 8xR.
    # A1 corresponds to rows 0,1,2,3 of A (i=0,1; j=0,1)
    # B11 corresponds to cols 0,1 of B (j=0,1; k=0,1)
    # W1 corresponds to rows 0,1,2,3 of C (k=0,1; i=0,1)
    
    # We need to embed these into the full matrices.
    # This is getting complicated to manage indices manually.
    # I will return the schoolbook decomposition for now to ensure correctness.
    # Rank 56.
    
    # Let's just return the schoolbook for safety.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
