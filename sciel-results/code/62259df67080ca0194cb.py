import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Direct construction: 4 independent Strassen 2x2x2 blocks
    # Each block handles a pair of column-blocks of A and row-blocks of B
    # producing a column-block of C

    # Strassen 7 products for a single 2x2x2 block
    # Indices within a block: a00,a01,a10,a11 for A; b00,b01,b10,b11 for B; c00,c01,c10,c11 for C

    # Block 0: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    # Block 1: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    # Block 2: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    # Block 3: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]

    blocks = [
        # (A_row_offsets, A_col_offsets, B_row_offsets, B_col_offsets, C_row_offsets, C_col_offsets)
        ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3]),
    ]

    for blk in range(4):
        ar0, ar1 = blocks[blk][0]
        ac0, ac1 = blocks[blk][1]
        br0, br1 = blocks[blk][2]
        bc0, bc1 = blocks[blk][3]
        cr0, cr1 = blocks[blk][4]
        cc0, cc1 = blocks[blk][5]

        base = blk * 7

        # A indices
        aa00 = ar0 * m + ac0
        aa01 = ar0 * m + ac1
        aa10 = ar1 * m + ac0
        aa11 = ar1 * m + ac1

        # B indices
        bb00 = br0 * p + bc0
        bb01 = br0 * p + bc1
        bb10 = br1 * p + bc0
        bb11 = br1 * p + bc1

        # C indices (W uses k*n + i where k=col, i=row)
        cc00 = cc0 * n + cr0
        cc01 = cc1 * n + cr0
        cc10 = cc0 * n + cr1
        cc11 = cc1 * n + cr1

        # M1 = (a00+a11)(b00+b11)
        U[aa00][base + 0] += 1
        U[aa11][base + 0] += 1
        V[bb00][base + 0] += 1
        V[bb11][base + 0] += 1
        W[cc00][base + 0] += 1
        W[cc11][base + 0] += 1

        # M2 = (a10+a11)b00
        U[aa10][base + 1] += 1
        U[aa11][base + 1] += 1
        V[bb00][base + 1] += 1
        W[cc10][base + 1] += 1
        W[cc11][base + 1] -= 1

        # M3 = a00(b01-b11)
        U[aa00][base + 2] += 1
        V[bb01][base + 2] += 1
        V[bb11][base + 2] -= 1
        W[cc01][base + 2] += 1
        W[cc11][base + 2] += 1

        # M4 = a11(b10-b00)
        U[aa11][base + 3] += 1
        V[bb10][base + 3] += 1
        V[bb00][base + 3] -= 1
        W[cc00][base + 3] += 1
        W[cc10][base + 3] += 1

        # M5 = (a00+a01)b11
        U[aa00][base + 4] += 1
        U[aa01][base + 4] += 1
        V[bb11][base + 4] += 1
        W[cc00][base + 4] -= 1
        W[cc01][base + 4] += 1

        # M6 = (a10-a00)(b00+b01)
        U[aa10][base + 5] += 1
        U[aa00][base + 5] -= 1
        V[bb00][base + 5] += 1
        V[bb01][base + 5] += 1
        W[cc11][base + 5] += 1

        # M7 = (a01-a11)(b10+b11)
        U[aa01][base + 6] += 1
        U[aa11][base + 6] -= 1
        V[bb10][base + 6] += 1
        V[bb11][base + 6] += 1
        W[cc00][base + 6] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
