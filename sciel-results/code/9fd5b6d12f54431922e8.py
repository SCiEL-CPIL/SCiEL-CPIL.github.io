"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank for 2x2x2 is 7
    # Block rank for 1x1x7 is 7
    # Total rank 7 * 7 = 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u[r][i][j], v[r][j][k'], w[r][i][k']
    # r in 0..6, i,j,k' in 0..1
    
    u = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    
    v = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    
    w = [
        [[1, 0], [0, 1]], # M1: C00+=1, C11+=1
        [[0, 0], [1, -1]],# M2: C10+=1, C11-=1
        [[0, 1], [0, 1]], # M3: C01+=1, C11+=1
        [[1, 0], [1, 0]], # M4: C00+=1, C10+=1
        [[-1, 1], [0, 0]],# M5: C00-=1, C01+=1
        [[0, 0], [0, 1]], # M6: C11+=1
        [[1, 0], [0, 0]]  # M7: C00+=1
    ]
    
    for r in range(7):
        for s in range(7):
            c = r * 7 + s
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = u[r][i][j]
            
            for j in range(m):
                for k in range(p):
                    # k = 2*s + k'
                    if k // 2 == s:
                        k_prime = k % 2
                        V[j * p + k][c] = v[r][j][k_prime]
                    else:
                        V[j * p + k][c] = 0
            
            for k in range(p):
                if k // 2 == s:
                    k_prime = k % 2
                    for i in range(n):
                        W[k * n + i][c] = w[r][i][k_prime]
                else:
                    for i in range(n):
                        W[k * n + i][c] = 0
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
