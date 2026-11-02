"""Tensor decomposition for 3x4x8 matrix multiplication using recursive Strassen."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88

    col = 0

    def add_trivial(i0, j0, k0, ni, nj, np):
        """Add trivial decomposition for sub-block [i0:i0+ni] x [j0:j0+nj] x [k0:k0+np]"""
        nonlocal col
        for di in range(ni):
            for dj in range(nj):
                for dk in range(np):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    def add_strassen(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block at offset (i0, j0, k0)"""
        nonlocal col
        # Strassen 2x2x2: 7 multiplications
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
        # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
        # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6

        # Map local (di, dj) to global U index, (dj, dk) to V, (dk, di) to W
        def u_idx(di, dj):
            return (i0 + di) * m + (j0 + dj)

        def v_idx(dj, dk):
            return (j0 + dj) * p + (k0 + dk)

        def w_idx(dk, di):
            return (k0 + dk) * n + (i0 + di)

        # M1 = (a+d)(e+h): U has +1 at (0,0),(1,1); V has +1 at (0,0),(0,1); W has +1 at all 4 positions
        U[u_idx(0,0)][col] = 1; U[u_idx(1,1)][col] = 1
        V[v_idx(0,0)][col] = 1; V[v_idx(0,1)][col] = 1
        W[w_idx(0,0)][col] = 1; W[w_idx(0,1)][col] = 1
        W[w_idx(1,0)][col] = 1; W[w_idx(1,1)][col] = 1
        col += 1

        # M2 = (c+d)e: U has +1 at (1,0),(1,1); V has +1 at (0,0); W has +1 at (0,1),(1,1)
        U[u_idx(1,0)][col] = 1; U[u_idx(1,1)][col] = 1
        V[v_idx(0,0)][col] = 1
        W[w_idx(0,1)][col] = 1; W[w_idx(1,1)][col] = 1
        col += 1

        # M3 = a(f-h): U has +1 at (0,0); V has +1 at (0,1), -1 at (1,1); W has +1 at (0,1),(1,1)
        U[u_idx(0,0)][col] = 1
        V[v_idx(0,1)][col] = 1; V[v_idx(1,1)][col] = -1
        W[w_idx(0,1)][col] = 1; W[w_idx(1,1)][col] = 1
        col += 1

        # M4 = d(g-e): U has +1 at (1,1); V has +1 at (1,0), -1 at (0,0); W has +1 at (0,0),(1,0)
        U[u_idx(1,1)][col] = 1
        V[v_idx(1,0)][col] = 1; V[v_idx(0,0)][col] = -1
        W[w_idx(0,0)][col] = 1; W[w_idx(1,0)][col] = 1
        col += 1

        # M5 = (a+b)h: U has +1 at (0,0),(0,1); V has +1 at (1,1); W has +1 at (0,1),(1,1)
        U[u_idx(0,0)][col] = 1; U[u_idx(0,1)][col] = 1
        V[v_idx(1,1)][col] = 1
        W[w_idx(0,1)][col] = 1; W[w_idx(1,1)][col] = 1
        col += 1

        # M6 = (c-a)(e+f): U has +1 at (1,0), -1 at (0,0); V has +1 at (0,0),(0,1); W has +1 at (1,0),(1,1)
        U[u_idx(1,0)][col] = 1; U[u_idx(0,0)][col] = -1
        V[v_idx(0,0)][col] = 1; V[v_idx(0,1)][col] = 1
        W[w_idx(1,0)][col] = 1; W[w_idx(1,1)][col] = 1
        col += 1

        # M7 = (b-d)(g+h): U has +1 at (0,1), -1 at (1,1); V has +1 at (1,0),(1,1); W has +1 at (0,0),(0,1)
        U[u_idx(0,1)][col] = 1; U[u_idx(1,1)][col] = -1
        V[v_idx(1,0)][col] = 1; V[v_idx(1,1)][col] = 1
        W[w_idx(0,0)][col] = 1; W[w_idx(0,1)][col] = 1
        col += 1

    # Rows 0,1 (2x4x8): split into two 2x2x8 blocks
    # Block A: j in [0,1], k in [0,7] -> 4 Strassen 2x2x2 blocks
    for k_pair in range(4):
        add_strassen(0, 0, k_pair * 2)
    # Block B: j in [2,3], k in [0,7] -> 4 Strassen 2x2x2 blocks
    for k_pair in range(4):
        add_strassen(0, 2, k_pair * 2)

    # Row 2 (1x4x8): trivial
    add_trivial(2, 0, 0, 1, 4, 8)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
