"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j=2 terms (10 terms)
    # A[:, 2] * B[2, :]
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2a: Strassen for j in {0,1}, k in {0,1} (7 terms)
    # A[0:2, 0:2] * B[0:2, 0:2]
    # U indices: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # V indices: 0,1,5,6 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    # W indices: 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # p1 = (A00 + A11)(B00 + B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # p2 = (A10 + A11)B00
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # p3 = A00(B01 - B11)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[1][col] = 1
    col += 1
    
    # p4 = A11(B10 - B00)
    U[3][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[2][col] = 1
    col += 1
    
    # p5 = (A00 + A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[1][col] = 1
    col += 1
    
    # p6 = (A10 - A00)(B00 + B01)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # p7 = (A01 - A11)(B10 + B11)
    U[1][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # Part 2b: Schoolbook for j in {0,1}, k in {2,3,4} (12 terms)
    # A[0:2, 0:2] * B[0:2, 2:5]
    for i in range(n):
        for j in range(2):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
