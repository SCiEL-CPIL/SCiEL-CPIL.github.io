"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    # Split middle dimension m=4 into 2+2
    # A: n x m = 3x4, split into A1 (3x2) and A2 (3x2)
    # B: m x p = 4x8, split into B1 (2x8) and B2 (2x8)
    # C = A1*B1 + A2*B2
    # Each product is rank n * 2 * p = 3*2*8 = 24
    # Total rank = 48
    
    rank = 2 * n * 2 * p  # 48
    U = [[0] * rank for _ in range(n * m)]  # 12 x 48
    V = [[0] * rank for _ in range(m * p)]  # 32 x 48
    W = [[0] * rank for _ in range(p * n)]  # 24 x 48
    
    col = 0
    # First block: A1 (rows 0-1 of A's columns) * B1 (rows 0-1 of B)
    for i in range(n):
        for j in range(2):  # j=0,1 for A1
            for k in range(p):  # k=0..7
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Second block: A2 (rows 2-3 of A's columns) * B2 (rows 2-3 of B)
    for i in range(n):
        for j in range(2, 4):  # j=2,3 for A2
            for k in range(p):  # k=0..7
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
