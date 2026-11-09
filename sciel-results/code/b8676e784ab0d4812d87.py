import time
import sys

DIMENSIONS = [2, 4, 8]

# Strassen coefficient forms for 2x2 * 2x2
# A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
# P1=(a00+a11)(b00+b11) P2=(a10+a11)b00 P3=a00(b01-b11) P4=a11(b10-b00)
# P5=(a00+a01)b11 P6=(a10-a00)(b00+b01) P7=(a01-a11)(b10+b11)
# C00=P1+P4-P5+P7  C01=P3+P5  C10=P2+P4  C11=P1-P2+P3+P6
X_FORMS = [
    [1, 0, 0, 1],   # P1
    [0, 0, 1, 1],   # P2
    [1, 0, 0, 0],   # P3
    [0, 0, 0, 1],   # P4
    [1, 1, 0, 0],   # P5
    [-1, 0, 1, 0],  # P6
    [0, 1, 0, -1],  # P7
]
Y_FORMS = [
    [1, 0, 0, 1],   # P1
    [1, 0, 0, 0],   # P2
    [0, 1, 0, -1],  # P3
    [-1, 0, 1, 0],  # P4
    [0, 0, 0, 1],   # P5
    [1, 1, 0, 0],   # P6
    [0, 0, 1, 1],   # P7
]
# W[k*n+i, r] = sum_s Z_combos[i][k'][s] * P_s
Z_COMBOS = [
    [1, 0, 0, 1, -1, 0, 1],   # C00 (i=0,k'=0)
    [0, 0, 1, 0, 1, 0, 0],    # C01 (i=0,k'=1)
    [0, 1, 0, 1, 0, 0, 0],    # C10 (i=1,k'=0)
    [1, -1, 1, 0, 0, 1, 0],   # C11 (i=1,k'=1)
]


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8

    # C = A @ B,  A: n x m (2x4), B: m x p (4x8), C: n x p (2x8)
    # Split A into two 2x2 column blocks (A_L cols 0-1, A_R cols 2-3)
    # and B into four 2x4 column groups (cols 0-1, 2-3, 4-5, 6-7).
    # For each group g and A-block b, the 2x2 x 2x2 core is handled by
    # Strassen (rank 7).  4 groups * 2 blocks * 7 = 56 factors.
    #
    # Indexing:
    #   U[i*m + j, r]  (i in [0,n), j in [0,m))
    #   V[j*p + k, r]  (j in [0,m), k in [0,p))
    #   W[k*n + i, r]  (k in [0,p), i in [0,n))  -- official trace convention

    rank = 4 * 2 * 7  # 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    r = 0
    for g in range(4):      # output column pair: cols 2g, 2g+1
        for b in range(2):  # A column block: cols 2b, 2b+1
            for s in range(7):
                # U: A element (i, 2b + jp)
                for i in range(n):
                    for jp in range(2):
                        U[i * m + (2 * b + jp)][r] = X_FORMS[s][i * 2 + jp]
                # V: B element (2b + jp, 2g + kp)
                for jp in range(2):
                    for kp in range(2):
                        V[(2 * b + jp) * p + (2 * g + kp)][r] = Y_FORMS[s][jp * 2 + kp]
                # W: C element (k=2g+kp, i)
                for i in range(n):
                    for kp in range(2):
                        W[(2 * g + kp) * n + i][r] = Z_COMBOS[i * 2 + kp][s]
                r += 1

    return {"U": U, "V": V, "W": W}
