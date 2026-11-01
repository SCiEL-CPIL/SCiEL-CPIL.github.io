"""Matrix multiplication 2x2 * 2x12 using rank-5 per 2x3 block (total rank 20)."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    # 4 blocks of width 3
    num_blocks = p // 3  # 4
    rank_per_block = 5
    rank = num_blocks * rank_per_block  # 20

    U = [[0] * rank for _ in range(n * m)]  # (4, 20)
    V = [[0] * rank for _ in range(m * p)]  # (24, 20)
    W = [[0] * rank for _ in range(p * n)]  # (24, 20)

    # Rank-5 decomposition for 2x2 * 2x3 -> 2x3
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12, B13], [B21, B22, B23]]
    # M1 = (A11+A22)(B11+B21)
    # M2 = (A11+A22)(B12+B22)
    # M3 = (A11+A22)(B13+B23)
    # M4 = A11*(B11+B21-B22)
    # M5 = A22*(B22-B11+B13)
    # C11 = M1 - M4
    # C12 = M2 - M4
    # C13 = M3 - M5
    # C21 = M4
    # C22 = M2 - M5
    # C23 = M5

    for b in range(num_blocks):
        base = b * rank_per_block
        # A indices: A11 -> row 0, A12 -> row 1, A21 -> row 2, A22 -> row 3
        # B row j (0 or 1), col k (0,1,2): index j*3 + k
        # C row r (0 or 1), col k (0,1,2): index k*n + r = k*2 + r

        # M1 = (A11+A22)(B11+B21)
        U[0][base + 0] += 1  # A11
        U[3][base + 0] += 1  # A22
        V[0][base + 0] += 1  # B11
        V[3][base + 0] += 1  # B21
        # C11 += M1: C[0*2+0] = C[0]
        W[0][base + 0] += 1
        # C21 += 0 for M1

        # M2 = (A11+A22)(B12+B22)
        U[0][base + 1] += 1  # A11
        U[3][base + 1] += 1  # A22
        V[1][base + 1] += 1  # B12
        V[4][base + 1] += 1  # B22
        # C12 += M2: C[1*2+0] = C[2]
        W[2][base + 1] += 1
        # C22 += M2: C[1*2+1] = C[3]
        W[3][base + 1] += 1

        # M3 = (A11+A22)(B13+B23)
        U[0][base + 2] += 1  # A11
        U[3][base + 2] += 1  # A22
        V[2][base + 2] += 1  # B13
        V[5][base + 2] += 1  # B23
        # C13 += M3: C[2*2+0] = C[4]
        W[4][base + 2] += 1

        # M4 = A11*(B11+B21-B22)
        U[0][base + 3] += 1  # A11
        V[0][base + 3] += 1  # B11
        V[3][base + 3] += 1  # B21
        V[4][base + 3] -= 1  # -B22
        # C11 -= M4: C[0]
        W[0][base + 3] -= 1
        # C12 -= M4: C[2]
        W[2][base + 3] -= 1
        # C21 += M4: C[1]
        W[1][base + 3] += 1

        # M5 = A22*(B22-B11+B13)
        U[3][base + 4] += 1  # A22
        V[4][base + 4] += 1  # B22
        V[0][base + 4] -= 1  # -B11
        V[2][base + 4] += 1  # B13
        # C13 -= M5: C[4]
        W[4][base + 4] -= 1
        # C22 -= M5: C[3]
        W[3][base + 4] -= 1
        # C23 += M5: C[5]
        W[5][base + 4] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
