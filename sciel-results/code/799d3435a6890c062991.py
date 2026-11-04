import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 77

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 steps: each step is (alpha_dict, beta_dict, gamma_dict)
    # Keys are (row_block, col_block) in {0,1}x{0,1}
    S7 = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    def idx_u(bi, bj, si, sj):
        return (bi*2+si)*m + (bj*2+sj)

    def idx_v(bi, bj, si, sj):
        return (bi*2+si)*p + (bj*2+sj)

    def idx_w(bi, bj, si, sj):
        return (bj*2+sj)*n + (bi*2+si)

    # Part 1: 4x4 * 4x4 -> 4x4 (columns 0..48)
    for sa in range(7):
        aa, ba, ga = S7[sa]
        for sb in range(7):
            ab, bb, gb = S7[sb]
            c = sa*7 + sb
            for (bi,bj), va in aa.items():
                for (si,sj), va2 in ab.items():
                    U[idx_u(bi,bj,si,sj)][c] += va*va2
            for (bi,bj), vb in ba.items():
                for (si,sj), vb2 in bb.items():
                    V[idx_v(bi,bj,si,sj)][c] += vb*vb2
            for (bi,bj), vc in ga.items():
                for (si,sj), vc2 in gb.items():
                    W[idx_w(bi,bj,si,sj)][c] += vc*vc2

    # Part 2: 4x4 * 4x2 -> 4x2 (columns 49..76)
    # B2 = B[:, 4:6] is 4x2, split into B_top (rows 0-1) and B_bot (rows 2-3)
    # C2 = C[:, 4:6] is 4x2, split into C_top (rows 0-1) and C_bot (rows 2-3)
    # Multiplications:
    #   A_top * B_top -> C_top
    #   A_bot * B_bot -> C_bot
    # where A_top = A[:2,:], A_bot = A[2:,:]

    # For A_top * B_top -> C_top (rows 0-1 of C, cols 4-5)
    # This is effectively: rows 0-1 of A (2x4) * rows 0-1 of B (2x2) -> rows 0-1 of C (2x2)
    # But we need to handle the 4x4 * 4x2 structure properly.
    # Actually, C[i,j] = sum_k A[i,k] * B[k,j]
    # For i in {0,1}, j in {4,5}: C[i,j] = sum_k A[i,k]*B[k,j]
    # = sum_{k=0}^{3} A[i,k]*B[k,j]
    # = sum_{k=0}^{1} A[i,k]*B[k,j] + sum_{k=2}^{3} A[i,k]*B[k,j]
    # = (A[:2,:] * B[:2, 4:6])[i, j-4] + (A[2:,:] * B[2:, 4:6])[i, j-4]

    # So C_top = A_top * B_top_2x2 + A_bot * B_bot_2x2
    # where A_top is 2x4, B_top_2x2 is 2x2, etc.
    # But this is a sum of two matrix products, not a single one.
    # We can't use Strassen directly on a 2x4 * 2x2 product in the same way.

    # Let me reconsider. The standard approach for 4x4 * 4x6:
    # Partition B into B1 (4x4) and B2 (4x2).
    # C1 = A * B1 (4x4) - handled by Part 1 with rank 49.
    # C2 = A * B2 (4x2) - need rank for this.

    # C2[i,j] = sum_k A[i,k] * B2[k,j] where j is 0 or 1 (local), global col 4+j.
    # This is a 4x4 * 4x2 multiplication.
    # We can view this as: partition A into 2x2 blocks A00,A01,A10,A11.
    # Partition B2 into B_top (2x2) and B_bot (2x2).
    # C2_top (2x2) = A00*B_top + A01*B_top ... wait no.
    # C2[0:2, 0:2] = sum_k A[0:2,k]*B2[k,0:2] = A[0:2,:]*B2[:,:]
    # = A00*B_top + A01*B_top? No, A[0:2,:] is 2x4, B2 is 4x2.
    # = [A00 A01] * [B_top; B_bot] = A00*B_top + A01*B_bot (each 2x2)
    # Similarly C2[2:4, 0:2] = A10*B_top + A11*B_bot

    # So we have 4 separate 2x2x2 multiplications:
    # A00*B_top, A01*B_bot, A10*B_top, A11*B_bot
    # Each can be done with Strassen rank 7.
    # Total rank 28.

    mults = [
        (0, 0, 0),  # A00 * B_top -> C2_top
        (0, 1, 0),  # A01 * B_bot -> C2_top
        (1, 0, 1),  # A10 * B_top -> C2_bot
        (1, 1, 1),  # A11 * B_bot -> C2_bot
    ]

    for mi, (ai, aj, ci) in enumerate(mults):
        for sb in range(7):
            ab, bb, gb = S7[sb]
            c = 49 + mi*7 + sb
            # A block (ai, aj): rows ai*2..ai*2+1, cols aj*2..aj*2+1
            for (si,sj), va in ab.items():
                U[idx_u(ai,aj,si,sj)][c] += va
            # B block: top if ci==0 else bot, cols 4-5
            b_row_start = 0 if ci == 0 else 2
            for (si,sj), vb in bb.items():
                # B row = b_row_start + si, col = 4 + sj
                r = (b_row_start+si)*p + (4+sj)
                V[r][c] += vb
            # C block: top if ci==0 else bot, cols 4-5
            c_row_start = 0 if ci == 0 else 2
            for (si,sj), vc in gb.items():
                # C row = c_row_start + si, col = 4 + sj
                # W index = col * n + row
                r = (4+sj)*n + (c_row_start+si)
                W[r][c] += vc

    return {"U": U, "V": V, "W": W}
