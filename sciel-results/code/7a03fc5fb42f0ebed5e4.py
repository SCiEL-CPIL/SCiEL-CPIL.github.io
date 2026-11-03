"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_indices(x_idx, y_idx, z_idx, col):
        # M1
        U[x_idx[0]][col] += 1
        U[x_idx[3]][col] += 1
        V[y_idx[0]][col] += 1
        V[y_idx[3]][col] += 1
        W[z_idx[0]][col] += 1
        W[z_idx[3]][col] += 1
        col += 1
        # M2
        U[x_idx[2]][col] += 1
        U[x_idx[3]][col] += 1
        V[y_idx[0]][col] += 1
        W[z_idx[2]][col] += 1
        W[z_idx[3]][col] += -1
        col += 1
        # M3
        U[x_idx[0]][col] += 1
        V[y_idx[2]][col] += 1
        V[y_idx[3]][col] += -1
        W[z_idx[1]][col] += 1
        W[z_idx[3]][col] += 1
        col += 1
        # M4
        U[x_idx[3]][col] += 1
        V[y_idx[1]][col] += 1
        V[y_idx[0]][col] += -1
        W[z_idx[0]][col] += 1
        W[z_idx[2]][col] += 1
        col += 1
        # M5
        U[x_idx[0]][col] += 1
        U[x_idx[1]][col] += 1
        V[y_idx[3]][col] += 1
        W[z_idx[0]][col] += -1
        W[z_idx[1]][col] += 1
        col += 1
        # M6
        U[x_idx[2]][col] += 1
        U[x_idx[0]][col] += -1
        V[y_idx[0]][col] += 1
        V[y_idx[1]][col] += 1
        W[z_idx[3]][col] += 1
        col += 1
        # M7
        U[x_idx[1]][col] += 1
        U[x_idx[3]][col] += -1
        V[y_idx[2]][col] += 1
        V[y_idx[3]][col] += 1
        W[z_idx[0]][col] += 1
        W[z_idx[1]][col] += -1
        col += 1
        return col

    col = 0
    for k in range(4):
        # Pk = AL * Btop_k
        x_idx = [0, 1, 2, 3]
        y_idx = [2*k, 2*k+1, 8+2*k, 8+2*k+1]
        z_idx = [4*k, 4*k+2, 4*k+1, 4*k+3]
        col = add_strassen_indices(x_idx, y_idx, z_idx, col)
        
        # Qk = AR * Bbot_k
        x_idx = [4, 5, 6, 7]
        y_idx = [16+2*k, 16+2*k+1, 24+2*k, 24+2*k+1]
        z_idx = [4*k, 4*k+2, 4*k+1, 4*k+3]
        col = add_strassen_indices(x_idx, y_idx, z_idx, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
