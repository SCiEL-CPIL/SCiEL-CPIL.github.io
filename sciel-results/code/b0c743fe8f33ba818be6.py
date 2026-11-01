DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A1 B1
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1.0
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
            
    def add_strassen(a_cols, b_rows, c_start):
        nonlocal col
        a1, a2 = a_cols
        b1, b2 = b_rows
        c1, c2 = c_start, c_start + 1
        
        u1 = 0 * m + a1
        u2 = 1 * m + a2
        u3 = 1 * m + a1
        u5 = 0 * m + a1
        u6 = 0 * m + a2
        
        v1 = b1 * p + c1
        v2 = b2 * p + c2
        v3 = b2 * p + c1
        v4 = b2 * p + c2
        v5 = b1 * p + c2
        
        w11 = c1 * n + 0
        w12 = c2 * n + 0
        w21 = c1 * n + 1
        w22 = c2 * n + 1
        
        # M1
        U[u1][col] += 1.0
        U[u2][col] += 1.0
        V[v1][col] += 1.0
        V[v2][col] += 1.0
        W[w11][col] += 1.0
        W[w22][col] += 1.0
        col += 1
        
        # M2
        U[u3][col] += 1.0
        U[u2][col] += 1.0
        V[v1][col] += 1.0
        W[w21][col] += 1.0
        col += 1
        
        # M3
        U[u1][col] += 1.0
        U[u2][col] += -1.0
        V[v3][col] += 1.0
        V[v4][col] += 1.0
        W[w22][col] += 1.0
        col += 1
        
        # M4
        U[u1][col] += 1.0
        V[v5][col] += 1.0
        V[v4][col] += -1.0
        W[w11][col] += 1.0
        W[w12][col] += 1.0
        col += 1
        
        # M5
        U[u2][col] += 1.0
        V[v1][col] += 1.0
        V[v5][col] += -1.0
        W[w21][col] += 1.0
        W[w11][col] += -1.0
        col += 1
        
        # M6
        U[u5][col] += 1.0
        U[u6][col] += 1.0
        V[v4][col] += 1.0
        W[w12][col] += 1.0
        W[w22][col] += 1.0
        col += 1
        
        # M7
        U[u3][col] += 1.0
        U[u1][col] += -1.0
        V[v1][col] += 1.0
        V[v5][col] += 1.0
        W[w11][col] += 1.0
        W[w21][col] += -1.0
        col += 1

    add_strassen([1, 2], [1, 2], 0)
    add_strassen([1, 2], [1, 2], 2)
    add_strassen([1, 2], [1, 2], 4)
    add_strassen([1, 2], [1, 2], 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
