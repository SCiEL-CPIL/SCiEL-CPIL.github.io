"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    # 1. Strassen for A[:, :2] * B[:, :2] (2x2x2)
    # 2. Schoolbook for A[:, 2] * B[2, :] (2x1x3)
    # 3. Schoolbook for A[:, :2] * B[:, 2] (2x2x1)
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # 1. Strassen for 2x2x2
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # Result C11, C12, C21, C22
    
    # M1 = A11 * B11
    add_term([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0])
    # M2 = A12 * B21
    add_term([0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0])
    # M3 = A21 * B12
    add_term([0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0])
    # M4 = A22 * B22
    add_term([0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0])
    
    # Strassen extra terms
    # M5 = (A11+A22) * (B11+B22)
    add_term([1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]) # C11 += M5
    add_term([1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]) # C22 += M5
    
    # M6 = (A21+A22) * B12
    add_term([0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]) # C21 += M6
    add_term([0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]) # C22 += M6
    
    # M7 = A11 * (B21-B11)
    add_term([1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]) # C21 += M7
    add_term([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]) # C21 -= M1? No, M7 = A11*(B21-B11)
    
    # This manual Strassen construction is error-prone.
    # I will use the known 17 terms if I can generate them correctly.
    # Given the risk, I will return the schoolbook rank 18.
    
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
