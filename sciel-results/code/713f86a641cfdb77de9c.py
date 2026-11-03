"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook decomposition
    def schoolbook(n, m, p):
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
        return U, V, W, rank
    
    # Strassen decomposition for 2x2 x 2x2
    def strassen_2x2():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2 x 2x2
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # C[0,0] += M1, C[1,1] += M1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # C[1,0] += M2
        
        # M3 = A[0,0] * (B[1,0] - B[1,1])
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1  # C[0,1] += M3
        
        # M4 = A[1,1] * (B[0,1] - B[0,0])
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1  # C[0,0] += M4, C[1,0] += M4
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[0][4] = -1  # C[0,1] += M5, C[0,0] -= M5
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[1,0])
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1  # C[1,1] += M6
        
        # M7 = (A[0,1] - A[1,1]) * (B[0,1] + B[1,1])
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1  # C[0,0] += M7
        
        return U, V, W, 7
    
    # For now, just return the schoolbook
    U, V, W, rank = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
