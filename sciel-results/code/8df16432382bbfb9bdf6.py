import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Strassen forms for 2x2x2
    # U[i*2+j'] for i in 0..1, j' in 0..1 (A is 2x2)
    # V[j'*2+k'] for j' in 0..1, k' in 0..1 (B is 2x2)
    # W[k'*2+i] for k' in 0..1, i in 0..1 (C is 2x2)
    #
    # Standard Strassen:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6

    U_strassen = [
        [1, 0, 0, 1],   # m1: A00 + A11
        [0, 0, 1, 1],   # m2: A10 + A11
        [1, 0, 0, 0],   # m3: A00
        [0, 0, 0, 1],   # m4: A11
        [1, 1, 0, 0],   # m5: A00 + A01
        [-1, 0, 1, 0],  # m6: A10 - A00
        [0, 1, 0, -1],  # m7: A01 - A11
    ]
    V_strassen = [
        [1, 0, 0, 1],   # m1: B00 + B11
        [1, 0, 0, 0],   # m2: B00
        [0, 1, 0, -1],  # m3: B01 - B11
        [-1, 0, 1, 0],  # m4: B10 - B00
        [0, 0, 0, 1],   # m5: B11
        [1, 1, 0, 0],   # m6: B00 + B10
        [0, 0, 1, 1],   # m7: B01 + B11
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 = m1 + m4 - m5 + m7
        [0, 0, 1, 0, 1, 0, 0],    # C01 = m3 + m5
        [0, 1, 0, 1, 0, 0, 0],    # C10 = m2 + m4
        [1, -1, 1, 0, 0, 1, 0],   # C11 = m1 - m2 + m3 + m6
    ]

    num_groups = 4      # g = 0,1,2,3 (k-pairs)
    num_ablocks = 2     # b = 0,1 (j-blocks)
    num_strassen = 7    # s = 0..6
    rank = num_groups * num_ablocks * num_strassen  # 56

    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                # U: A is 2x2, indexed by i (row) and jp (col within block)
                # A index = i*m + b*2 + jp = i*4 + 2*b + jp
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][r] = U_strassen[s][i * 2 + jp]

                # V: B is 2x2, indexed by jp (row) and kp (col within block)
                # B index = (b*2 + jp)*p + (2*g + kp)
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r] = V_strassen[s][jp * 2 + kp]

                # W: C is 2x2, indexed by kp (col) and i (row)
                # C index = (2*g + kp)*n + i
                for i in range(n):
                    for kp in range(2):
                        c_idx = (2 * g + kp) * n + i
                        W[c_idx][r] = W_strassen[i * 2 + kp][s]

                r += 1

    return {"U": U, "V": V, "W": W}
