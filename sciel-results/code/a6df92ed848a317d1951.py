"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    # 6 pairs of columns, each using Strassen rank-7
    num_pairs = p // 2  # 6
    rank = 7 * num_pairs  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (24, 42)
    
    for j in range(num_pairs):
        base = 7 * j
        col2j = 2 * j
        col2j1 = 2 * j + 1
        
        # M1_j: (a00+a11)(b0_{2j}+b0_{2j+1}+b1_{2j}+b1_{2j+1})
        c = base + 0
        U[0][c] += 1   # a00
        U[3][c] += 1   # a11
        V[col2j][c] += 1       # b0_{2j}
        V[col2j1][c] += 1      # b0_{2j+1}
        V[m + col2j][c] += 1   # b1_{2j}
        V[m + col2j1][c] += 1  # b1_{2j+1}
        W[col2j * n + 0][c] += 1    # C[0, 2j]
        W[col2j1 * n + 1][c] += 1   # C[1, 2j+1]
        
        # M2_j: (a10+a11)b0_{2j}
        c = base + 1
        U[2][c] += 1   # a10
        U[3][c] += 1   # a11
        V[col2j][c] += 1  # b0_{2j}
        W[col2j * n + 1][c] += 1    # C[1, 2j]
        W[col2j1 * n + 1][c] += -1  # C[1, 2j+1]
        
        # M3_j: a00(b0_{2j+1}-b1_{2j+1})
        c = base + 2
        U[0][c] += 1   # a00
        V[col2j1][c] += 1       # b0_{2j+1}
        V[m + col2j1][c] += -1  # b1_{2j+1}
        W[col2j1 * n + 0][c] += 1  # C[0, 2j+1]
        W[col2j1 * n + 1][c] += 1  # C[1, 2j+1]
        
        # M4_j: a11(b1_{2j}-b0_{2j})
        c = base + 3
        U[3][c] += 1   # a11
        V[m + col2j][c] += 1    # b1_{2j}
        V[col2j][c] += -1       # b0_{2j}
        W[col2j * n + 0][c] += 1  # C[0, 2j]
        W[col2j * n + 1][c] += 1  # C[1, 2j]
        
        # M5_j: (a00+a01)b1_{2j+1}
        c = base + 4
        U[0][c] += 1   # a00
        U[1][c] += 1   # a01
        V[m + col2j1][c] += 1  # b1_{2j+1}
        W[col2j * n + 0][c] += -1   # C[0, 2j]
        W[col2j1 * n + 0][c] += 1   # C[0, 2j+1]
        
        # M6_j: (a10-a00)(b0_{2j}+b0_{2j+1})
        c = base + 5
        U[2][c] += 1    # a10
        U[0][c] += -1   # a00
        V[col2j][c] += 1     # b0_{2j}
        V[col2j1][c] += 1    # b0_{2j+1}
        W[col2j1 * n + 1][c] += 1  # C[1, 2j+1]
        
        # M7_j: (a01-a11)(b1_{2j}+b1_{2j+1})
        c = base + 6
        U[1][c] += 1    # a01
        U[3][c] += -1   # a11
        V[m + col2j][c] += 1    # b1_{2j}
        V[m + col2j1][c] += 1   # b1_{2j+1}
        W[col2j * n + 0][c] += 1  # C[0, 2j]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
