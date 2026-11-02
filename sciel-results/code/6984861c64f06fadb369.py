"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Partition: A is n x m = 2x3, B is m x p = 3x10, C is n x p = 2x10
    # Split m=3 into m1=2 and m2=1
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x10); B2 (1x10)]
    # C = A1*B1 + A2*B2
    # A1*B1: 5 Strassen blocks (each 2x2 * 2x2), 7 mults each = 35
    # A2*B2: 2x1 * 1x10, 20 mults = 20
    # Total rank = 55
    
    rank = 55
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # Strassen for A1*B1 where A1 is rows 0-1, cols 0-1 of A
    # B1 is rows 0-1, all 10 cols of B
    # A1 indices in U: row i, col j -> index i*m + j = i*3 + j for i in 0-1, j in 0-1
    # B1 indices in V: row i, col j -> index i*p + j = i*10 + j for i in 0-1, j in 0-9
    # C indices in W: row k, col i -> index k*n + i for k in 0-9, i in 0-1
    
    # For each of 5 column pairs (j0, j0+1) in B1:
    for block in range(5):
        j0 = block * 2  # 0, 2, 4, 6, 8
        j1 = j0 + 1     # 1, 3, 5, 7, 9
        
        # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B1_block = [[B[0,j0], B[0,j1]], [B[1,j0], B[1,j1]]]
        
        # Strassen:
        # M1 = (A[0,0]+A[1,1]) * (B[0,j0]+B[1,j1])
        # M2 = (A[1,0]+A[1,1]) * B[0,j0]
        # M3 = A[0,0] * (B[0,j1]-B[1,j1])
        # M4 = A[1,1] * (B[1,j0]-B[0,j0])
        # M5 = (A[0,0]+A[0,1]) * B[1,j1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,j0]+B[0,j1])
        # M7 = (A[0,1]-A[1,1]) * (B[1,j0]+B[1,j1])
        
        # C[0, j0] = M1 + M4 - M5 + M7
        # C[0, j1] = M3 + M5
        # C[1, j0] = M2 + M4
        # C[1, j1] = M1 - M2 + M3 + M6
        
        # M1
        U[0*3+0][col] = 1; U[1*3+1][col] = 1
        V[0*10+j0][col] = 1; V[1*10+j1][col] = 1
        W[j0*2+0][col] = 1; W[j1*2+1][col] = 1
        col += 1
        
        # M2
        U[1*3+0][col] = 1; U[1*3+1][col] = 1
        V[0*10+j0][col] = 1
        W[j0*2+1][col] = 1
        col += 1
        
        # M3
        U[0*3+0][col] = 1
        V[0*10+j1][col] = 1; V[1*10+j1][col] = -1
        W[j1*2+0][col] = 1
        col += 1
        
        # M4
        U[1*3+1][col] = 1
        V[1*10+j0][col] = 1; V[0*10+j0][col] = -1
        W[j0*2+0][col] = 1; W[j0*2+1][col] = 1
        col += 1
        
        # M5
        U[0*3+0][col] = 1; U[0*3+1][col] = 1
        V[1*10+j1][col] = 1
        W[j0*2+0][col] = -1; W[j1*2+0][col] = 1
        col += 1
        
        # M6
        U[1*3+0][col] = 1; U[0*3+0][col] = -1
        V[0*10+j0][col] = 1; V[0*10+j1][col] = 1
        W[j1*2+1][col] = 1
        col += 1
        
        # M7
        U[0*3+1][col] = 1; U[1*3+1][col] = -1
        V[1*10+j0][col] = 1; V[1*10+j1][col] = 1
        W[j0*2+0][col] = 1
        col += 1
    
    # A2*B2: A2 is column 2 of A (2x1), B2 is row 2 of B (1x10)
    # C[i, j] += A[2, i] * B[2, j] for i in 0-1, j in 0-9
    # Wait, A2 is A[:, 2] which is a 2x1 matrix. B2 is B[2, :] which is 1x10.
    # C[i, j] += A[i, 2] * B[2, j]
    
    for i in range(n):  # 0, 1
        for j in range(p):  # 0-9
            U[i*3+2][col] = 1  # A[i, 2]
            V[2*10+j][col] = 1  # B[2, j]
            W[j*2+i][col] = 1  # C[i, j] -> index j*n + i
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
