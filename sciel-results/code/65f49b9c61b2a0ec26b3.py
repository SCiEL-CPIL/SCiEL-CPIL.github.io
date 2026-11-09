"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # 2x2x2 Strassen rank-7 decomposition
    # U: 4x7, rows (i,j) for i,j in {0,1}
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ]
    # V: 4x7, rows (j,k) for j,k in {0,1}
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ]
    # W: 4x7, rows (k,i) for k,i in {0,1}
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ]

    # Build a 4x4x4 rank-21 decomposition by tiling 2x2x2 Strassen blocks.
    # For each (bi,bj,bk) in {0,1}^3 we place 7 rank-1 terms.
    nb2 = 2  # 2x2 blocks in each dim for 4x4x4
    total_rank_444 = nb2 * nb2 * nb2 * 7  # 56 -> but we can do better with known 21

    # Actually, the 4x4x4 tensor has rank <= 21 (known result).
    # Constructing the exact 21-rank decomposition is complex. 
    # Instead, let's use the 2x2x2 tiling which gives rank 56 for 4x4x4,
    # then for 4x4x6 we tile in the p dimension: 4x4x6 = two 4x4x3 blocks.
    # Each 4x4x3 = three 4x4x1... no, that's schoolbook.
    #
    # Let's stick with the proven 2x2x2 tiling for 4x4x6.
    # n=4, m=4, p=6
    # We tile into 2x2 blocks in i and j, and 2x2 blocks in k (with last block being 2x2 for k=4,5)
    # So: 2 blocks in i, 2 blocks in j, 3 blocks in k (k=0,1; k=2,3; k=4,5)
    # Total rank = 2*2*3*7 = 84. This is what the current program does.
    #
    # Can we do better? The rank of 4x4x6 matrix multiplication tensor.
    # Schoolbook: 4*4*6 = 96.
    # 2x2x2 tiling: 84.
    # 
    # Let me try a different approach: use the fact that 4x4x6 = 4x4x4 + 4x4x2.
    # Rank(4x4x4) <= 21 (known), Rank(4x4x2) <= 10 (can be shown).
    # Total: 21 + 10 = 31? That seems too optimistic and I don't have the explicit factors.
    #
    # Let me try to construct a better decomposition by using 2x2x3 blocks.
    # For a 2x2x3 matrix multiplication (A[2x2] * B[2x3] = C[2x3]):
    # Schoolbook: 2*2*3 = 12.
    # Can we do better? Let's use Strassen-like ideas.
    # Actually, let me just use the 2x2x2 tiling but with a twist.
    #
    # Alternative: use 4x4x4 rank-21 for the first 4 columns of p, and 4x4x2 for the last 2.
    # I need explicit 4x4x4 rank-21 and 4x4x2 rank-10 decompositions.
    #
    # Since I don't have those memorized, let me try a different strategy:
    # Use a 4x4x4 decomposition with rank 35 (which I can construct from 2x2x2 tiling + sharing)
    # and 4x4x2 with rank 16 (2x2x2 tiling). Total: 35+16=51? No, 2x2x2 tiling for 4x4x4 is 56.
    #
    # Let me just keep the rank-84 solution but write it more cleanly.
    # Actually, let me try to find if there's a rank-77 or similar by using
    # the fact that the last 2x2x2 block can share some structure.
    #
    # For now, I'll construct the rank-84 solution carefully.

    n_blocks = n // 2  # 2
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 3
    block_size = 2

    total_rank = n_blocks * m_blocks * p_blocks * 7
    # total_rank = 84

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j][col] = S_U[di * 2 + dj][c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k][col] = S_V[dj * 2 + dk][c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i][col] = S_W[dk * 2 + di][c]
                col_offset += 7

    if time.time() > deadline:
        # Should not happen, but return what we have
        return {"U": U, "V": W, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
