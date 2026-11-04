"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 28

    def strassen_2x2(A_rows, B_rows, B_cols, W_i_indices, base):
        a0, a1 = A_rows
        b0, b1 = B_rows
        k0, k1 = B_cols
        w0, w1 = W_i_indices

        col = base
        # M1
        U[a0*m+b0][col] += 1.0
        U[a1*m+b1][col] += 1.0
        V[b0*p+k0][col] += 1.0
        V[b1*p+k1][col] += 1.0
        W[k0*n+w0][col] += 1.0
        W[k1*n+w1][col] += 1.0
        col += 1
        # M2
        U[a1*m+b0][col] += 1.0
        U[a1*m+b1][col] += 1.0
        V[b0*p+k0][col] += 1.0
        W[k0*n+w1][col] += 1.0
        col += 1
        # M3
        U[a0*m+b0][col] += 1.0
        V[b1*p+k0][col] += 1.0
        V[b1*p+k1][col] -= 1.0
        W[k1*n+w0][col] += 1.0
        col += 1
        # M4
        U[a1*m+b1][col] += 1.0
        V[b0*p+k1][col] += 1.0
        V[b0*p+k0][col] -= 1.0
        W[k0*n+w0][col] -= 1.0
        W[k0*n+w1][col] += 1.0
        col += 1
        # M5
        U[a0*m+b0][col] += 1.0
        U[a0*m+b1][col] += 1.0
        V[b1*p+k1][col] += 1.0
        W[k0*n+w0][col] += 1.0
        W[k1*n+w0][col] += 1.0
        col += 1
        # M6
        U[a1*m+b0][col] += 1.0
        U[a0*m+b0][col] -= 1.0
        V[b0*p+k0][col] += 1.0
        V[b0*p+k1][col] += 1.0
        W[k0*n+w0][col] += 1.0
        col += 1
        # M7
        U[a0*m+b1][col] += 1.0
        U[a1*m+b1][col] -= 1.0
        V[b1*p+k0][col] += 1.0
        V[b1*p+k1][col] += 1.0
        W[k1*n+w1][col] += 1.0
        col += 1
        return base + 7

    # Left block: C[:, 0:2] = A1*B11 + A2*B21
    col = strassen_2x2([0,1], [0,1], [0,1], [0,1], 0)
    col = strassen_2x2([0,1], [2,3], [0,1], [0,1], col)
    # Right block: C[:, 2:4] = A1*B12 + A2*B22
    col = strassen_2x2([0,1], [0,1], [2,3], [0,1], col)
    col = strassen_2x2([0,1], [2,3], [2,3], [0,1], col)

    assert col == rank

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
