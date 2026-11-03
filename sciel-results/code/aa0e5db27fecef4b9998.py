def _build_rank30():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 30

    U = [[0] * rank for _ in range(n * m)]  # 8 x 30
    V = [[0] * rank for _ in range(m * p)]  # 20 x 30
    W = [[0] * rank for _ in range(p * n)]  # 10 x 30

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        # ... same as before ...
        col = c
        return c

    def add_rank1_2x2x1(i0, j0, k0, col_start):
        # Adds a rank-1 term for 2x2 matrix times 2x1 vector
        # A is 2x2 (rows i0, i0+1, cols j0, j0+1)
        # B is 2x1 (rows j0, j0+1, col k0)
        # C is 2x1 (rows i0, i0+1, col k0)
        # U: 2x2 all 1s
        # V: 2x1 all 1s
        # W: 1x2 all 1s
        c = col_start
        U[i0 * m + j0][c] = 1
        U[i0 * m + (j0 + 1)][c] = 1
        U[(i0 + 1) * m + j0][c] = 1
        U[(i0 + 1) * m + (j0 + 1)][c] = 1
        
        V[j0 * p + k0][c] = 1
        V[(j0 + 1) * p + k0][c] = 1
        
        W[k0 * n + i0][c] = 1
        W[k0 * n + (i0 + 1)][c] = 1
        
        col = c + 1
        return col

    # 4 Strassen blocks
    col = add_strassen_222(0, 0, 0, col)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    
    # 2 Rank-1 terms for col 4
    # A rows 0,1 (i0=0), B rows 0,1 (j0=0), B col 4 (k0=4)
    col = add_rank1_2x2x1(0, 0, 4, col)
    # A rows 0,1 (i0=0), B rows 2,3 (j0=2), B col 4 (k0=4)
    col = add_rank1_2x2x1(0, 2, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W, rank
