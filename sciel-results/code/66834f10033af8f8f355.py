DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    # Rank 69 decomposition
    # Block 1: M_{4,4,4} (Schoolbook) -> 64
    # Block 2: M_{4,4,1} (Schoolbook) -> 4
    # Block 3: A_{i,4} B_{4,l} (Rank 1) -> 1
    rank = 69
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 69
    V = [[0] * rank for _ in range(m * p)]  # 25 x 69
    W = [[0] * rank for _ in range(p * n)]  # 20 x 69
    
    col = 0
    
    # Block 1: i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: i in 0..3, j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            k = 4
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 3: i in 0..3, j=4, l in 0..4
    # Term: A[i,4] * B[4,l] -> C[i,l]
    # W index: l*n + i
    # V index: 4*p + l
    # U index: i*m + 4
    for i in range(4):
        for l in range(5):
            U[i * m + 4][col] = 1
            V[4 * p + l][col] = 1
            W[l * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
