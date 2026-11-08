import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[10 + k1][col] += 1
        W[2 * k0][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[2 * k0 + 1][col] += 1
        W[2 * k1 + 1][col] -= 1
        col += 1
        
        U[0][col] += 1
        V[k1][col] += 1
        V[10 + k1][col] -= 1
        W[2 * k1][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        U[4][col] += 1
        V[10 + k0][col] += 1
        V[k0][col] -= 1
        W[2 * k0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k1][col] += 1
        W[2 * k0][col] -= 1
        W[2 * k1][col] += 1
        col += 1
        
        U[3][col] += 1
        U[0][col] -= 1
        V[k0][col] += 1
        V[k1][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        U[1][col] += 1
        U[4][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k1][col] += 1
        W[2 * k0][col] += 1
        col += 1
        
        U[2][col] += 1
        V[20 + k0][col] += 1
        W[2 * k0][col] += 1
        col += 1
        
        U[2][col] += 1
        V[20 + k1][col] += 1
        W[2 * k1][col] += 1
        col += 1
        
        U[5][col] += 1
        V[20 + k0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
        U[5][col] += 1
        V[20 + k1][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
