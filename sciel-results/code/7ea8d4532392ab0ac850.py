"""Program that finds a rank-55 tensor decomposition for 2x3 x 3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6x55
    V = [[0] * rank for _ in range(m * p)]  # 30x55
    W = [[0] * rank for _ in range(p * n)]  # 20x55
    
    col = 0
    # Strassen for A[0:2, 0:2] * B[0:2, :] (5 pairs of columns)
    for pair in range(5):
        k0 = 2 * pair
        k1 = 2 * pair + 1
        
        # M1 = (A00+A11)(B00+B11) -> C00+C11
        U[0][col] = 1   # A[0,0]
        U[4][col] = 1   # A[1,1]
        V[0 * 10 + k0][col] = 1   # B[0, k0]
        V[1 * 10 + k1][col] = 1   # B[1, k1]
        W[k0 * 2 + 0][col] = 1   # C[0, k0]
        W[k1 * 2 + 1][col] = 1   # C[1, k1]
        col += 1
        
        # M2 = (A10+A11)B00 -> C10
        U[3][col] = 1   # A[1,0]
        U[4][col] = 1   # A[1,1]
        V[0 * 10 + k0][col] = 1   # B[0, k0]
        W[k0 * 2 + 1][col] = 1   # C[1, k0]
        col += 1
        
        # M3 = A00(B01-B11) -> C01
        U[0][col] = 1   # A[0,0]
        V[0 * 10 + k1][col] = 1   # B[0, k1]
        V[1 * 10 + k1][col] = -1  # B[1, k1]
        W[k1 * 2 + 0][col] = 1   # C[0, k1]
        col += 1
        
        # M4 = A11(B10-B00) -> C10
        U[4][col] = 1   # A[1,1]
        V[1 * 10 + k0][col] = 1   # B[1, k0]
        V[0 * 10 + k0][col] = -1  # B[0, k0]
        W[k0 * 2 + 1][col] = 1   # C[1, k0]
        col += 1
        
        # M5 = (A00+A10)B11 -> C01 - C00
        U[0][col] = 1   # A[0,0]
        U[1][col] = 1   # A[0,1]
        V[1 * 10 + k1][col] = 1   # B[1, k1]
        W[k1 * 2 + 0][col] = 1   # C[0, k1]
        W[k0 * 2 + 0][col] = -1  # C[0, k0]
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        U[3][col] = 1   # A[1,0]
        U[0][col] = -1  # A[0,0]
        V[0 * 10 + k0][col] = 1   # B[0, k0]
        V[0 * 10 + k1][col] = 1   # B[0, k1]
        W[k1 * 2 + 1][col] = 1   # C[1, k1]
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00
        U[1][col] = 1   # A[0,1]
        U[4][col] = -1  # A[1,1]
        V[1 * 10 + k0][col] = 1   # B[1, k0]
        V[1 * 10 + k1][col] = 1   # B[1, k1]
        W[k0 * 2 + 0][col] = 1   # C[0, k0]
        col += 1
    
    # Schoolbook for A[:,2] * B[2,:]
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][col] = 1   # A[i, 2]
            V[2 * 10 + k][col] = 1   # B[2, k]
            W[k * 2 + i][col] = 1   # C[i, k]
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
