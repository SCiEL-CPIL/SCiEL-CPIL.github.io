import time
import sys
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    t_limit = 115.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8

    # Strassen 2x2x2 decomposition (rank 7)
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]

    # r=0 (M1) = (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1

    # r=1 (M2) = (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1

    # r=2 (M3) = a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1

    # r=3 (M4) = a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1

    # r=4 (M5) = (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1

    # r=5 (M6) = (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1

    # r=6 (M7) = (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1

    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            a_map = [0]*4
            if s == 0:
                a_map[0] = 0*4 + 0; a_map[1] = 0*4 + 1
                a_map[2] = 1*4 + 0; a_map[3] = 1*4 + 1
            else:
                a_map[0] = 0*4 + 2; a_map[1] = 0*4 + 3
                a_map[2] = 1*4 + 2; a_map[3] = 1*4 + 3
            b_map = [0]*4
            if s == 0:
                b_map[0] = 0*8 + (2*j); b_map[1] = 0*8 + (2*j+1)
                b_map[2] = 1*8 + (2*j); b_map[3] = 1*8 + (2*j+1)
            else:
                b_map[0] = 2*8 + (2*j); b_map[1] = 2*8 + (2*j+1)
                b_map[2] = 3*8 + (2*j); b_map[3] = 3*8 + (2*j+1)
            c_map = [0]*4
            c_map[0] = (2*j) * 2 + 0
            c_map[1] = (2*j+1) * 2 + 0
            c_map[2] = (2*j) * 2 + 1
            c_map[3] = (2*j+1) * 2 + 1
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val

    if time.time() - start_time > t_limit:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
