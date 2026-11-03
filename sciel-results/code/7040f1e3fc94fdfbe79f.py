"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition:
    # Split m=5 into m1=2, m2=3
    # Split p=7 into p1=3, p2=4
    
    # Block 1: A1 (3x2) * B11 (2x3) -> C11 (3x3)
    # Block 2: A1 (3x2) * B12 (2x4) -> C12 (3x4)
    # Block 3: A2 (3x3) * B21 (3x3) -> C21 (3x3)
    # Block 4: A2 (3x3) * B22 (3x4) -> C22 (3x4)
    
    # Helper to create Strassen 2x2x2 decomposition
    def strassen_222():
        # Returns U, V, W for 2x2x2 multiplication
        # U: 4x7, V: 4x7, W: 4x7
        U = [
            [1, 0, 1, 0, 0, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V = [
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [1, 0, -1, -1, 1, 0, 1]
        ]
        # W rows correspond to C indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
        # Note: Indexing in W is k*n + i. n=2.
        # c00: 0*2+0=0
        # c10: 0*2+1=1
        # c01: 1*2+0=2
        # c11: 1*2+1=3
        W = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        return U, V, W

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to decompose n x m x p using Strassen where possible
    def decompose(n, m, p):
        # Base cases
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
        if n == 1 and m == 1 and p == 1:
            return ([[1]], [[1]], [[1]])
        
        # Split strategy:
        # If any dimension is 1, use schoolbook or split the others.
        # If all are >= 2, try to split into 2 and rest.
        
        # For 3x3x3, split n=3 into 1+2.
        if n == 3 and m == 3 and p == 3:
            # C = A1 B1 + A2 B2
            # A1: 1x3, A2: 2x3
            # B1: 3x3, B2: 3x3
            # Block 1: 1x3x3
            # Block 2: 2x3x3
            
            # Decompose 1x3x3
            # Split m=3 into 1+2
            # 1x1x3 + 1x2x3
            # 1x1x3: Schoolbook (3)
            # 1x2x3: Split p=3 into 1+2
            # 1x2x1 + 1x2x2
            # 1x2x1: Schoolbook (2)
            # 1x2x2: Split m=2? No. Split n=1? No.
            # 1x2x2: Schoolbook (4) or Strassen? No, n=1.
            
            # This is getting complicated. Let's just use Schoolbook for 3x3x3 if Strassen doesn't fit easily.
            # Rank 27.
            return schoolbook(3, 3, 3)

        # For 3x2x3
        # Split n=3 into 1+2
        # 1x2x3 + 2x2x3
        # 1x2x3: Schoolbook (6)
        # 2x2x3: Split p=3 into 1+2
        # 2x2x1 + 2x2x2
        # 2x2x1: Schoolbook (4)
        # 2x2x2: Strassen (7)
        # Total: 6 + 4 + 7 = 17
        
        if n == 3 and m == 2 and p == 3:
            U1, V1, W1 = schoolbook(1, 2, 3) # 1x2x3
            U2, V2, W2 = schoolbook(2, 2, 1) # 2x2x1
            U3, V3, W3 = strassen_222()      # 2x2x2
            
            # Assemble
            # Block 1: 1x2x3 (n=1, m=2, p=3)
            # Block 2: 2x2x1 (n=2, m=2, p=1)
            # Block 3: 2x2x2 (n=2, m=2, p=2)
            
            # Total n=3, m=2, p=3
            total_rank = len(U1[0]) + len(U2[0]) + len(U3[0])
            U = [[0] * total_rank for _ in range(3 * 2)]
            V = [[0] * total_rank for _ in range(2 * 3)]
            W = [[0] * total_rank for _ in range(3 * 3)]
            
            col = 0
            # Block 1: rows 0 of A, all m, all p
            for i in range(1):
                for j in range(2):
                    u_row = i * 2 + j
                    for c in range(len(U1[0])):
                        U[u_row][col + c] = U1[i * 2 + j][c]
            for j in range(2):
                for k in range(3):
                    v_row = j * 3 + k
                    for c in range(len(V1[0])):
                        V[v_row][col + c] = V1[j * 3 + k][c]
            for k in range(3):
                for i in range(1):
                    w_row = k * 3 + i
                    for c in range(len(W1[0])):
                        W[w_row][col + c] = W1[k * 1 + i][c]
            col += len(U1[0])
            
            # Block 2: rows 1-2 of A, all m, p=1 (first col of B)
            for i in range(2):
                for j in range(2):
                    u_row = (i + 1) * 2 + j
                    for c in range(len(U2[0])):
                        U[u_row][col + c] = U2[i * 2 + j][c]
            for j in range(2):
                for k in range(1):
                    v_row = j * 3 + k
                    for c in range(len(V2[0])):
                        V[v_row][col + c] = V2[j * 1 + k][c]
            for k in range(1):
                for i in range(2):
                    w_row = k * 3 + (i + 1)
                    for c in range(len(W2[0])):
                        W[w_row][col + c] = W2[k * 2 + i][c]
            col += len(U2[0])
            
            # Block 3: rows 1-2 of A, all m, p=2 (cols 1-2 of B)
            for i in range(2):
                for j in range(2):
                    u_row = (i + 1) * 2 + j
                    for c in range(len(U3[0])):
                        U[u_row][col + c] = U3[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    v_row = j * 3 + (k + 1)
                    for c in range(len(V3[0])):
                        V[v_row][col + c] = V3[j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    w_row = (k + 1) * 3 + (i + 1)
                    for c in range(len(W3[0])):
                        W[w_row][col + c] = W3[k * 2 + i][c]
                        
            return U, V, W

        # For 3x2x4
        # Split n=3 into 1+2
        # 1x2x4 + 2x2x4
        # 1x2x4: Schoolbook (8)
        # 2x2x4: Split p=4 into 2+2
        # 2x2x2 + 2x2x2
        # 7 + 7 = 14
        # Total: 8 + 14 = 22
        
        if n == 3 and m == 2 and p == 4:
            U1, V1, W1 = schoolbook(1, 2, 4)
            U2, V2, W2 = strassen_222()
            U3, V3, W3 = strassen_222()
            
            total_rank = len(U1[0]) + len(U2[0]) + len(U3[0])
            U = [[0] * total_rank for _ in range(3 * 2)]
            V = [[0] * total_rank for _ in range(2 * 4)]
            W = [[0] * total_rank for _ in range(4 * 3)]
            
            col = 0
            # Block 1: 1x2x4
            for i in range(1):
                for j in range(2):
                    u_row = i * 2 + j
                    for c in range(len(U1[0])):
                        U[u_row][col + c] = U1[i * 2 + j][c]
            for j in range(2):
                for k in range(4):
                    v_row = j * 4 + k
                    for c in range(len(V1[0])):
                        V[v_row][col + c] = V1[j * 4 + k][c]
            for k in range(4):
                for i in range(1):
                    w_row = k * 3 + i
                    for c in range(len(W1[0])):
                        W[w_row][col + c] = W1[k * 1 + i][c]
            col += len(U1[0])
            
            # Block 2: 2x2x2 (cols 0-1 of B)
            for i in range(2):
                for j in range(2):
                    u_row = (i + 1) * 2 + j
                    for c in range(len(U2[0])):
                        U[u_row][col + c] = U2[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    v_row = j * 4 + k
                    for c in range(len(V2[0])):
                        V[v_row][col + c] = V2[j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    w_row = k * 3 + (i + 1)
                    for c in range(len(W2[0])):
                        W[w_row][col + c] = W2[k * 2 + i][c]
            col += len(U2[0])
            
            # Block 3: 2x2x2 (cols 2-3 of B)
            for i in range(2):
                for j in range(2):
                    u_row = (i + 1) * 2 + j
                    for c in range(len(U3[0])):
                        U[u_row][col + c] = U3[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    v_row = j * 4 + (k + 2)
                    for c in range(len(V3[0])):
                        V[v_row][col + c] = V3[j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    w_row = (k + 2) * 3 + (i + 1)
                    for c in range(len(W3[0])):
                        W[w_row][col + c] = W3[k * 2 + i][c]
                        
            return U, V, W

        # For 3x3x4
        # Split p=4 into 2+2
        # 3x3x2 + 3x3x2
        # 3x3x2: Split n=3 into 1+2
        # 1x3x2 + 2x3x2
        # 1x3x2: Schoolbook (6)
        # 2x3x2: Split m=3 into 1+2
        # 2x1x2 + 2x2x2
        # 2x1x2: Schoolbook (4)
        # 2x2x2: Strassen (7)
        # Total 3x3x2: 6 + 4 + 7 = 17
        # Total 3x3x4: 17 + 17 = 34
        
        if n == 3 and m == 3 and p == 4:
            # We need to build 3x3x2 first
            def decompose_332():
                U1, V1, W1 = schoolbook(1, 3, 2)
                U2, V2, W2 = schoolbook(2, 1, 2)
                U3, V3, W3 = strassen_222()
                
                total_rank = len(U1[0]) + len(U2[0]) + len(U3[0])
                U = [[0] * total_rank for _ in range(3 * 3)]
                V = [[0] * total_rank for _ in range(3 * 2)]
                W = [[0] * total_rank for _ in range(2 * 3)]
                
                col = 0
                # Block 1: 1x3x2
                for i in range(1):
                    for j in range(3):
                        u_row = i * 3 + j
                        for c in range(len(U1[0])):
                            U[u_row][col + c] = U1[i * 3 + j][c]
                for j in range(3):
                    for k in range(2):
                        v_row = j * 2 + k
                        for c in range(len(V1[0])):
                            V[v_row][col + c] = V1[j * 2 + k][c]
                for k in range(2):
                    for i in range(1):
                        w_row = k * 3 + i
                        for c in range(len(W1[0])):
                            W[w_row][col + c] = W1[k * 1 + i][c]
                col += len(U1[0])
                
                # Block 2: 2x1x2 (rows 1-2 of A, col 0 of B)
                for i in range(2):
                    for j in range(1):
                        u_row = (i + 1) * 3 + j
                        for c in range(len(U2[0])):
                            U[u_row][col + c] = U2[i * 1 + j][c]
                for j in range(1):
                    for k in range(2):
                        v_row = j * 2 + k
                        for c in range(len(V2[0])):
                            V[v_row][col + c] = V2[j * 2 + k][c]
                for k in range(2):
                    for i in range(2):
                        w_row = k * 3 + (i + 1)
                        for c in range(len(W2[0])):
                            W[w_row][col + c] = W2[k * 2 + i][c]
                col += len(U2[0])
                
                # Block 3: 2x2x2 (rows 1-2 of A, cols 1-2 of B)
                for i in range(2):
                    for j in range(2):
                        u_row = (i + 1) * 3 + (j + 1)
                        for c in range(len(U3[0])):
                            U[u_row][col + c] = U3[i * 2 + j][c]
                for j in range(2):
                    for k in range(2):
                        v_row = (j + 1) * 2 + k
                        for c in range(len(V3[0])):
                            V[v_row][col + c] = V3[j * 2 + k][c]
                for k in range(2):
                    for i in range(2):
                        w_row = k * 3 + (i + 1)
                        for c in range(len(W3[0])):
                            W[w_row][col + c] = W3[k * 2 + i][c]
                            
                return U, V, W

            U1, V1, W1 = decompose_332()
            U2, V2, W2 = decompose_332()
            
            total_rank = len(U1[0]) + len(U2[0])
            U = [[0] * total_rank for _ in range(3 * 3)]
            V = [[0] * total_rank for _ in range(3 * 4)]
            W = [[0] * total_rank for _ in range(4 * 3)]
            
            col = 0
            # Block 1: 3x3x2 (cols 0-1 of B)
            for i in range(3):
                for j in range(3):
                    u_row = i * 3 + j
                    for c in range(len(U1[0])):
                        U[u_row][col + c] = U1[i * 3 + j][c]
            for j in range(3):
                for k in range(2):
                    v_row = j * 4 + k
                    for c in range(len(V1[0])):
                        V[v_row][col + c] = V1[j * 2 + k][c]
            for k in range(2):
                for i in range(3):
                    w_row = k * 3 + i
                    for c in range(len(W1[0])):
                        W[w_row][col + c] = W1[k * 3 + i][c]
            col += len(U1[0])
            
            # Block 2: 3x3x2 (cols 2-3 of B)
            for i in range(3):
                for j in range(3):
                    u_row = i * 3 + j
                    for c in range(len(U2[0])):
                        U[u_row][col + c] = U2[i * 3 + j][c]
            for j in range(3):
                for k in range(2):
                    v_row = j * 4 + (k + 2)
                    for c in range(len(V2[0])):
                        V[v_row][col + c] = V2[j * 2 + k][c]
            for k in range(2):
                for i in range(3):
                    w_row = (k + 2) * 3 + i
                    for c in range(len(W2[0])):
                        W[w_row][col + c] = W2[k * 3 + i][c]
                        
            return U, V, W

        # Fallback
        return schoolbook(n, m, p)

    # Main decomposition
    # Split m=5 into 2+3
    # Block 1: 3x2x3
    # Block 2: 3x2x4
    # Block 3: 3x3x3
    # Block 4: 3x3x4
    
    U1, V1, W1 = decompose(3, 2, 3)
    U2, V2, W2 = decompose(3, 2, 4)
    U3, V3, W3 = decompose(3, 3, 3)
    U4, V4, W4 = decompose(3, 3, 4)
    
    total_rank = len(U1[0]) + len(U2[0]) + len(U3[0]) + len(U4[0])
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A1 (3x2), B11 (2x3), C11 (3x3)
    for i in range(3):
        for j in range(2):
            u_row = i * 5 + j
            for c in range(len(U1[0])):
                U[u_row][col + c] = U1[i * 2 + j][c]
    for j in range(2):
        for k in range(3):
            v_row = j * 7 + k
            for c in range(len(V1[0])):
                V[v_row][col + c] = V1[j * 3 + k][c]
    for k in range(3):
        for i in range(3):
            w_row = k * 3 + i
            for c in range(len(W1[0])):
                W[w_row][col + c] = W1[k * 3 + i][c]
    col += len(U1[0])
    
    # Block 2: A1 (3x2), B12 (2x4), C12 (3x4)
    for i in range(3):
        for j in range(2):
            u_row = i * 5 + j
            for c in range(len(U2[0])):
                U[u_row][col + c] = U2[i * 2 + j][c]
    for j in range(2):
        for k in range(4):
            v_row = j * 7 + (k + 3)
            for c in range(len(V2[0])):
                V[v_row][col + c] = V2[j * 4 + k][c]
    for k in range(4):
        for i in range(3):
            w_row = (k + 3) * 3 + i
            for c in range(len(W2[0])):
                W[w_row][col + c] = W2[k * 3 + i][c]
    col += len(U2[0])
    
    # Block 3: A2 (3x3), B21 (3x3), C21 (3x3)
    for i in range(3):
        for j in range(3):
            u_row = i * 5 + (j + 2)
            for c in range(len(U3[0])):
                U[u_row][col + c] = U3[i * 3 + j][c]
    for j in range(3):
        for k in range(3):
            v_row = (j + 2) * 7 + k
            for c in range(len(V3[0])):
                V[v_row][col + c] = V3[j * 3 + k][c]
    for k in range(3):
        for i in range(3):
            w_row = k * 3 + i
            for c in range(len(W3[0])):
                W[w_row][col + c] = W3[k * 3 + i][c]
    col += len(U3[0])
    
    # Block 4: A2 (3x3), B22 (3x4), C22 (3x4)
    for i in range(3):
        for j in range(3):
            u_row = i * 5 + (j + 2)
            for c in range(len(U4[0])):
                U[u_row][col + c] = U4[i * 3 + j][c]
    for j in range(3):
        for k in range(4):
            v_row = (j + 2) * 7 + (k + 3)
            for c in range(len(V4[0])):
                V[v_row][col + c] = V4[j * 4 + k][c]
    for k in range(4):
        for i in range(3):
            w_row = (k + 3) * 3 + i
            for c in range(len(W4[0])):
                W[w_row][col + c] = W4[k * 3 + i][c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
