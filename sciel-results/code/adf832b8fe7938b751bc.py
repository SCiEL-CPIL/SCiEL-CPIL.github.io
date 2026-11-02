"""Matrix multiplication 6x6x6 rank-minimal decomposition.

Uses the Kronecker product of the 3x3x3 schoolbook tensor (rank 27) and the
2x2x2 Strassen tensor (rank 7), giving a 6x6x6 multiplication tensor of rank
27*7 = 189, which is the known best exact rank for 6x6 matrix multiplication.
"""
import time

DIMENSIONS = [6, 6, 6]
_DEADLINE = 140.0
_T0 = time.time()


def _deadline_exceeded():
    return (time.time() - _T0) > _DEADLINE


# EVOLVE-BLOCK-START
def construct():
    # ---- 3x3x3 schoolbook tensor (rank 27) ----
    U3 = [[0] * 27 for _ in range(9)]
    V3 = [[0] * 27 for _ in range(9)]
    W3 = [[0] * 27 for _ in range(9)]
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j][c] = 1
                V3[j * 3 + k][c] = 1
                W3[k * 3 + i][c] = 1
                c += 1

    # ---- 2x2x2 Strassen tensor (rank 7) ----
    # A idx (i,j): A11=(0,0),A12=(0,1),A21=(1,0),A22=(1,1)
    # B idx (j,k): B11=(0,0),B12=(0,1),B21=(1,0),B22=(1,1)
    # C idx (k,i): C11=(0,0),C12=(0,1),C21=(1,0),C22=(1,1)
    terms = [
        # M1=(A11-A22)(B21+B22) -> C11,C22
        ([((0, 0), (1, 0), 1), ((0, 0), (1, 1), 1),
          ((1, 1), (1, 0), -1), ((1, 1), (1, 1), -1)], [(0, 0), (1, 1)]),
        # M2=(A11+A12)B22 -> C12
        ([((0, 0), (1, 1), 1), ((0, 1), (1, 1), 1)], [(0, 1)]),
        # M3=(A21+A22)B11 -> C21
        ([((1, 0), (0, 0), 1), ((1, 1), (0, 0), 1)], [(1, 0)]),
        # M4=(A22-A11)(B11+B12) -> C11
        ([((1, 1), (0, 0), 1), ((1, 1), (0, 1), 1),
          ((0, 0), (0, 0), -1), ((0, 0), (0, 1), -1)], [(0, 0)]),
        # M5=(A11+A22)(B11+B22) -> C11,C22
        ([((0, 0), (0, 0), 1), ((0, 0), (1, 1), 1),
          ((1, 1), (0, 0), 1), ((1, 1), (1, 1), 1)], [(0, 0), (1, 1)]),
        # M6=(A11-A12)(B22+B12) -> C12,C22
        ([((0, 0), (1, 1), 1), ((0, 0), (0, 1), 1),
          ((0, 1), (1, 1), -1), ((0, 1), (0, 1), -1)], [(0, 1), (1, 1)]),
        # M7=(A22-A21)(B11+B21) -> C11,C21
        ([((1, 1), (0, 0), 1), ((1, 1), (1, 0), 1),
          ((1, 0), (0, 0), -1), ((1, 0), (1, 0), -1)], [(0, 0), (1, 0)]),
    ]

    U2 = [[0] * 7 for _ in range(4)]
    V2 = [[0] * 7 for _ in range(4)]
    W2 = [[0] * 7 for _ in range(4)]
    for t, (ab, cs) in enumerate(terms):
        for (ai, bj, coeff) in ab:
            i, j = ai
            jj, k = bj
            U2[i * 2 + j][t] += coeff
            V2[jj * 2 + k][t] += coeff
        for (k, i) in cs:
            W2[k * 2 + i][t] += 1

    # ---- Kronecker product -> 6x6x6, rank 189 ----
    rank = 27 * 7
    Ufull = [[0] * rank for _ in range(36)]
    Vfull = [[0] * rank for _ in range(36)]
    Wfull = [[0] * rank for _ in range(36)]

    for c3 in range(27):
        for c2 in range(7):
            col = c3 * 7 + c2
            for i3 in range(3):
                for j3 in range(3):
                    u3 = U3[i3 * 3 + j3][c3]
                    if not u3:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2 * 2 + j2][c2]
                            if not u2:
                                continue
                            Ufull[(i3 * 2 + i2) * 6 + (j3 * 2 + j2)][col] += u3 * u2

            for j3 in range(3):
                for k3 in range(3):
                    v3 = V3[j3 * 3 + k3][c3]
                    if not v3:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = V2[j2 * 2 + k2][c2]
                            if not v2:
                                continue
                            Vfull[(j3 * 2 + j2) * 6 + (k3 * 2 + k2)][col] += v3 * v2

            for k3 in range(3):
                for i3 in range(3):
                    w3 = W3[k3 * 3 + i3][c3]
                    if not w3:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = W2[k2 * 2 + i2][c2]
                            if not w2:
                                continue
                            Wfull[(k3 * 2 + k2) * 6 + (i3 * 2 + i2)][col] += w3 * w2

    return {"U": Ufull, "V": Vfull, "W": Wfull}
# EVOLVE-BLOCK-END
