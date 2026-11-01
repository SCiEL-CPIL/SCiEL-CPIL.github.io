"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(rank_offset, k0):
        c = rank_offset + 0
        U[0][c] = 1; U[3][c] = 1
        V[k0][c] = 1; V[5+k0+1][c] = 1
        W[2*k0][c] = 1; W[2*k0+3][c] = -1
        
        c = rank_offset + 1
        U[2][c] = 1; U[3][c] = 1
        V[k0][c] = 1
        W[2*k0+1][c] = 1; W[2*k0+3][c] = -1
        
        c = rank_offset + 2
        U[0][c] = 1
        V[5+k0][c] = 1; V[5+k0+1][c] = -1
        W[2*k0+2][c] = 1; W[2*k0+3][c] = 1
        
        c = rank_offset + 3
        U[3][c] = 1
        V[k0+1][c] = 1; V[5+k0][c] = -1
        W[2*k0][c] = -1; W[2*k0+2][c] = 1; W[2*k0+3][c] = -1
        
        c = rank_offset + 4
        U[0][c] = 1; U[1][c] = 1
        V[5+k0+1][c] = 1
        W[2*k0][c] = 1; W[2*k0+3][c] = 1
        
        c = rank_offset + 5
        U[2][c] = 1; U[0][c] = -1
        V[k0][c] = 1; V[5+k0][c] = 1
        W[2*k0+1][c] = 1
        
        c = rank_offset + 6
        U[1][c] = 1; U[3][c] = -1
        V[k0+1][c] = 1; V[5+k0+1][c] = 1
        W[2*k0][c] = -1; W[2*k0+3][c] = 1
    
    add_strassen(0, 0)
    add_strassen(7, 2)
    
    c = 14
    for i in range(2):
        for j in range(2):
            U[i*2+j][c] = 1
            V[j*5+4][c] = 1
            W[8+i][c] = 1
    
    if time.time() - start_time > 140:
        # Fallback to schoolbook if somehow taking too long (shouldn't happen)
        rank = 20
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
