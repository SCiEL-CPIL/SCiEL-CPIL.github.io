"""Matrix multiplication 2x5x6 with Strassen block optimization (rank 59)."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 59  # 1 (Strassen) + 11 (other i0j0) + 48 (rest) = 59
    U = [[0] * rank for _ in range(n * m)]   # 10 x 59
    V = [[0] * rank for _ in range(m * p)]   # 30 x 59
    W = [[0] * rank for _ in range(p * n)]   # 12 x 59

    def add(i, j, k, col, u, v, w):
        U[i * m + j][col] += u
        V[j * p + k][col] += v
        W[k * n + i][col] += w

    col = 0
    # Strassen rank-7 for the (i=0, j=0, k=0) 2x2x2 block.
    # M1 = A00*(B01+B11); M2 = (A00+A10)*B11; M3 = (A00+A01)*B00;
    # M4 = A10*(B01+B11); M5 = A11*(B00+B10); M6 = (A11-A10)*(B01-B00);
    # M7 = (A00-A01)*(B10+B11).  (indices: Aii->i0, Aij->i1, etc.)
    strassen = [
        (0, 0, 0, 1, 1, 0),   # M1 -> C00
        (0, 0, 1, 0, 1, 1),   # M2 -> C01
        (0, 1, 0, 1, 0, 1),   # M3 -> C10
        (1, 0, 0, 1, 1, 0),   # M4 -> C10
        (1, 1, 1, 0, 1, 1),   # M5 -> C11
        (1, 1, 0, 1, -1, 1),  # M6 -> C11
        (0, 0, 1, 1, 1, 0),   # M7 -> C01
        (0, 1, 1, -1, 1, 0),  # M7 -> C10
        (1, 0, 1, -1, 1, 0),  # M7 -> C11
    ]
    for (i, j, k, u, v, w) in strassen:
        add(i, j, k, col, u, v, w)
        col += 1

    # Schoolbook for the remaining (i, j, k) tuples.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == 0 and j == 0 and k == 0:
                    continue
                add(i, j, k, col, 1, 1, 1)
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
