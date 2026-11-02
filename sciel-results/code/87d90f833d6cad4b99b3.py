"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    for u in range(6):
        base_v = 2 * u
        base_w = 2 * u * n
        for t in range(7):
            c = u * 7 + t
            for i in range(4):
                U[i][c] = U_forms[t][i]
            
            # V updates
            if t == 0:
                V[base_v][c] += 1
                V[p + base_v + 1][c] += 1
            elif t == 1:
                V[base_v][c] += 1
            elif t == 2:
                V[base_v + 1][c] += 1
                V[p + base_v + 1][c] -= 1
            elif t == 3:
                V[p + base_v][c] += 1
                V[base_v][c] -= 1
            elif t == 4:
                V[p + base_v + 1][c] += 1
            elif t == 5:
                V[base_v][c] += 1
                V[base_v + 1][c] += 1
            elif t == 6:
                V[p + base_v][c] += 1
                V[p + base_v + 1][c] += 1
                
            # W updates
            if t == 0:
                W[base_w + 0][c] += 1
                W[base_w + n + 1][c] += 1
            elif t == 1:
                W[base_w + 1][c] += 1
                W[base_w + n + 1][c] -= 1
            elif t == 2:
                W[base_w + n + 0][c] += 1
                W[base_w + n + 1][c] += 1
            elif t == 3:
                W[base_w + 0][c] += 1
                W[base_w + 1][c] += 1
            elif t == 4:
                W[base_w + 0][c] -= 1
                W[base_w + n + 0][c] += 1
            elif t == 5:
                W[base_w + n + 1][c] += 1
            elif t == 6:
                W[base_w + 0][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
