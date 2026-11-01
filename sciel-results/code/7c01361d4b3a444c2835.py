"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    R = 49
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # 2x2 Strassen base factors (indices 0..6)
    U2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W2 = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # 4x4 = 2x2 blocks, each 2x2
    # A indices: (i,j) -> i*4 + j
    # B indices: (j,k) -> j*4 + k
    # C indices: (i,k) -> k*4 + i

    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for b1 in range(4):
                for b2 in range(4):
                    ca = U2[t1][b1]
                    cb = V2[t2][b2]
                    cc = W2[t1][b2]
                    if ca != 0 and cb != 0 and cc != 0:
                        for l1 in range(4):
                            for l2 in range(4):
                                ia = b1 * 2 + l1
                                ja = b1 * 2 + l2
                                ik = b2 * 2 + l2
                                jk = b2 * 2 + l1
                                idx_a = ia * 4 + ja
                                idx_b = ja * 4 + jk
                                idx_c = ik * 4 + ia
                                u = ca * U2[t1][l1]
                                v = cb * V2[t2][l1]
                                w = cc * W2[t1][l2]
                                U[idx_a][col] += u
                                V[idx_b][col] += v
                                W[idx_c][col] += w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
