"""Matrix multiplication 2x4x6 rank-minimal decomposition via Strassen tiling."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 6 * 7  # six 2x2x2 blocks, each rank 7 (Strassen)
    U = [[0] * rank for _ in range(n * m)]   # 8 x rank
    V = [[0] * rank for _ in range(m * p)]   # 24 x rank
    W = [[0] * rank for _ in range(p * n)]   # 12 x rank

    # Strassen 2x2x2 factors (entries in {0, ±1})
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11),
    # M4=A11(B10-B00), M5=(A00+A10)B11, M6=(A10-A00)(B00+B10),
    # M7=(A01-A11)(B01+B11)
    # Product entries: C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    Uf = [
        [1, 0, 1, 0, 1, 0, 0],  # A00
        [0, 1, 1, 0, 0, 1, -1], # A01
        [1, 1, 0, 0, 1, -1, 0], # A10
        [1, 0, 0, 1, 0, 0, -1], # A11
    ]
    Vf = [
        [1, 1, 0, 0, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 0, 1],  # B01
        [1, 0, -1, -1, 1, 1, 0],# B10
        [1, 0, -1, 0, 1, 0, 1], # B11
    ]
    Wf = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]

    # Tile the 2x4x6 product into six 2x2x2 blocks.
    # Block (a,b): A[:, 2b:2b+2] (2x2) * B[2b:2b+2, 2a:2a+2] (2x2) -> C[:, 2a:2a+2]
    blocks = []
    for a in range(p // 2):   # 0..2 (output column pair)
        for b in range(m // 2):  # 0..1 (middle index pair)
            blocks.append((a, b))

    col = 0
    for (a, b) in blocks:
        for li in range(4):
            i = li // 2          # 0/1 (output row within block)
            li2 = li % 2         # 0/1 (input A row within block)
            for lj in range(4):
                j = lj // 2      # 0/1 (input B col within block)
                lj2 = lj % 2     # 0/1 (input B row within block)
                for lk in range(4):
                    k = lk // 2  # 0/1 (output B col within block)
                    lk2 = lk % 2 # 0/1 (output C row within block)
                    u = Uf[li][lk]
                    v = Vf[lj][lk]
                    w = Wf[lk][li]
                    if u:
                        U[i * m + (2 * b + j)][col] += u
                    if v:
                        V[(2 * b + j) * p + (2 * a + k)][col] += v
                    if w:
                        W[(2 * a + k) * n + i][col] += w
        col += 7
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

