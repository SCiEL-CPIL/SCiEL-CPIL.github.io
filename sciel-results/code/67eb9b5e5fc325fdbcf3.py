"""Matrix multiplication 2x3x9 - rank-27 block decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose C = A * B where A is 2x3, B is 3x9.
    # B has 9 columns, which we view as 3 blocks of 3 columns each.
    # C[:, 3t:r] = A * B[:, 3t:3t+3] for t = 0, 1, 2.
    # Each block multiplication is a 2x3x3 tensor of rank 9.
    # Since the A-factors can be shared across blocks (they depend only on i,j),
    # the total rank is 3 * 9 = 27.
    
    R = 27
    U = [[0] * R for _ in range(n * m)]   # 6 rows
    V = [[0] * R for _ in range(m * p)]   # 27 rows
    W = [[0] * R for _ in range(p * n)]   # 18 rows
    
    col = 0
    for t in range(3):
        for j in range(3):
            for i in range(2):
                for r in range(3):
                    # U[i*m + j] = A_{i,j}
                    U[i * m + j][col] = 1
                    # V[j*p + (3*t + r)] = B_{j, 3t+r}
                    V[j * p + (3 * t + r)][col] = 1
                    # W[(3*t + r)*n + i] = C_{3t+r, i}
                    W[(3 * t + r) * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
