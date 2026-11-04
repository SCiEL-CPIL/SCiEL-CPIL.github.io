import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120.0

    n, m, p = DIMENSIONS  # 6, 6, 6
    # Each matrix is divided into 3x3 blocks of 2x2 sub-blocks
    S = 3  # blocks per row/col of the 2x2 sub-block grid
    s = 2  # size of each sub-block

    # Strassen's 7 products for 2x2 block multiplication:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each block (br,bc) refers to row-block br, col-block bc (0 or 1)
    # strassen_defs[term] = (left_list, right_list)
    # left_list: list of (br, bc, coeff) for A
    # right_list: list of (br, bc, coeff) for B

    strassen_defs = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]

    rank = S * S * 7  # 63
    U = [[0] * rank for _ in range(n * m)]  # (36, 63)
    V = [[0] * rank for _ in range(m * p)]  # (36, 63)
    W = [[0] * rank for _ in range(p * n)]  # (36, 63)

    col = 0
    for bi in range(S):   # block row of A (i-direction)
        for bj in range(S):  # block col of A (j-direction) = block row of B
            for term in range(7):
                left_list, right_list = strassen_defs[term]

                # Fill U: rows (bi*2 + br, bj*2 + bc) for A block (bi,bj)
                for (br, bc, coeff) in left_list:
                    row_idx = (bi * s + br) * s + bc  # row in A = i*6+j
                    U[row_idx][col] += coeff

                # Fill V: rows (bj*2 + br, (bi+? ) ... )
                # B block at position (bj, bk) where bk = bi (since C block row = bi, col = bj)
                # Actually B's block row = bj, block col = bi (since C = A*B, C's block (bi,bj) needs A block (bi,*) and B block (*,bj) with * = bi for this term's position)
                # B block (br, bc) in the 2x2 sub-block: row-block br of B = bj*2+br, col-block = bi*2+bc
                for (br, bc, coeff) in right_list:
                    row_idx = (bj * s + br) * s + (bi * s + bc)  # row in B = j*6+k
                    V[row_idx][col] += coeff

                # Fill W: C block (bi, bj), sub-block (br, bc)
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # W index: (p*n) = k*6+i, so for C block (bi,bj): k = bj*2+br, i = bi*2+bc
                W_out = [
                    [(0,0,1), (1,1,1), (0,1,-1), (1,0,1)],   # M1: C00
                    [(1,0,1)],                                  # M2: C10
                    [(0,1,1), (1,1,1)],                         # M3: C01
                    [(1,1,1), (1,0,1)],                         # M4: C10
                    [(0,0,-1), (0,1,1)],                        # M5: C01, C00
                    [(1,1,1)],                                  # M6: C11
                    [(0,0,1), (1,1,1)],                         # M7: C11, C00
                ]
                for (br, bc, coeff) in W_out[term]:
                    row_idx = (bj * s + br) * s + (bi * s + bc)  # row in C = k*6+i
                    W[row_idx][col] += coeff

                col += 1

    if time.time() > deadline - 10.0:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
