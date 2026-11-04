"""Matrix multiplication 6x6x6 - Strassen-based decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm on 2x2 blocks within a 3x3 block structure.
    # Treat 6x6 as 3x3 blocks of 2x2 matrices.
    # For each block multiplication, use Strassen (rank 7).
    # Total rank = 27 * 7 = 189.
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for s in range(3):
            for j in range(3):
                for k in range(7):
                    # A block entries: a=(2i,2s), b=(2i,2s+1), c=(2i+1,2s), d=(2i+1,2s+1)
                    idx_a = (2 * i) * 6 + (2 * s)
                    idx_b = (2 * i) * 6 + (2 * s + 1)
                    idx_c = (2 * i + 1) * 6 + (2 * s)
                    idx_d = (2 * i + 1) * 6 + (2 * s + 1)
                    
                    # B block entries: e=(2s,2j), f=(2s,2j+1), g=(2s+1,2j), h=(2s+1,2j+1)
                    idx_e = (2 * s) * 6 + (2 * j)
                    idx_f = (2 * s) * 6 + (2 * j + 1)
                    idx_g = (2 * s + 1) * 6 + (2 * j)
                    idx_h = (2 * s + 1) * 6 + (2 * j + 1)
                    
                    # C block entries (W index = col*6 + row):
                    idx_w00 = (2 * j) * 6 + (2 * i)
                    idx_w01 = (2 * j + 1) * 6 + (2 * i)
                    idx_w10 = (2 * j) * 6 + (2 * i + 1)
                    idx_w11 = (2 * j + 1) * 6 + (2 * i + 1)
                    
                    # U coefficients
                    if k == 0:
                        U[idx_a][col] = 1
                        U[idx_d][col] = 1
                    elif k == 1:
                        U[idx_c][col] = 1
                        U[idx_d][col] = 1
                    elif k == 2:
                        U[idx_a][col] = 1
                    elif k == 3:
                        U[idx_d][col] = 1
                    elif k == 4:
                        U[idx_a][col] = 1
                        U[idx_b][col] = 1
                    elif k == 5:
                        U[idx_c][col] = 1
                        U[idx_a][col] = -1
                    elif k == 6:
                        U[idx_b][col] = 1
                        U[idx_d][col] = -1
                    
                    # V coefficients
                    if k == 0:
                        V[idx_e][col] = 1
                        V[idx_h][col] = 1
                    elif k == 1:
                        V[idx_e][col] = 1
                    elif k == 2:
                        V[idx_f][col] = 1
                        V[idx_h][col] = -1
                    elif k == 3:
                        V[idx_g][col] = 1
                        V[idx_e][col] = -1
                    elif k == 4:
                        V[idx_h][col] = 1
                    elif k == 5:
                        V[idx_e][col] = 1
                        V[idx_f][col] = 1
                    elif k == 6:
                        V[idx_g][col] = 1
                        V[idx_h][col] = 1
                    
                    # W coefficients
                    if k == 0:
                        W[idx_w00][col] = 1
                        W[idx_w11][col] = 1
                    elif k == 1:
                        W[idx_w10][col] = 1
                        W[idx_w11][col] = -1
                    elif k == 2:
                        W[idx_w01][col] = 1
                        W[idx_w11][col] = 1
                    elif k == 3:
                        W[idx_w00][col] = 1
                        W[idx_w10][col] = 1
                    elif k == 4:
                        W[idx_w00][col] = -1
                        W[idx_w01][col] = 1
                    elif k == 5:
                        W[idx_w11][col] = 1
                    elif k == 6:
                        W[idx_w00][col] = 1
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
