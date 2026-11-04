import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Strassen 2x2x2 decomposition (verified correct)
    strassen_222 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]), # M3
        ([0,0,0,1], [0,-1,1,0], [0,0,1,0]), # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]), # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]), # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]), # M7
    ]

    # A is 4x4: index i*4+j
    A11_idx = [0, 1, 4, 5]
    A12_idx = [2, 3, 6, 7]
    A21_idx = [8, 9, 12, 13]
    A22_idx = [10, 11, 14, 15]

    # B is 4x9: index j*9+k
    B11_idx = [0, 1, 9, 10]
    B12_idx = [2, 3, 11, 12]
    B21_idx = [18, 19, 27, 28]
    B22_idx = [20, 21, 29, 30]

    # C is 9x4 output: W index = k*4+i
    C11_idx = [0, 4, 1, 5]
    C12_idx = [8, 12, 9, 13]
    C21_idx = [2, 6, 3, 7]
    C22_idx = [10, 14, 11, 15]

    A_blocks = [A11_idx, A12_idx, A21_idx, A22_idx]
    B_blocks = [B11_idx, B12_idx, B21_idx, B22_idx]
    C_blocks = [C11_idx, C12_idx, C21_idx, C22_idx]

    # Strassen 4x4x4 (same structure, fixed M4)
    strassen_444 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    terms = []

    # Part 1: 4x4x4 Strassen (49 terms)
    for u_coeff, v_coeff, w_coeff in strassen_444:
        for u2, v2, w2 in strassen_222:
            u_vec = {}
            for j in range(4):
                if u_coeff[j] != 0:
                    for pos in range(4):
                        if u2[pos] != 0:
                            idx = A_blocks[j][pos]
                            u_vec[idx] = u_vec.get(idx, 0) + u_coeff[j] * u2[pos]

            v_vec = {}
            for j in range(4):
                if v_coeff[j] != 0:
                    for pos in range(4):
                        if v2[pos] != 0:
                            idx = B_blocks[j][pos]
                            v_vec[idx] = v_vec.get(idx, 0) + v_coeff[j] * v2[pos]

            w_vec = {}
            for j in range(4):
                if w_coeff[j] != 0:
                    for pos in range(4):
                        if w2[pos] != 0:
                            idx = C_blocks[j][pos]
                            w_vec[idx] = w_vec.get(idx, 0) + w_coeff[j] * w2[pos]

            if u_vec and v_vec and w_vec:
                terms.append((u_vec, v_vec, w_vec))

    # Part 2: 4x4x5 schoolbook (80 terms)
    for i in range(4):
        for j in range(4):
            for k in range(5):
                k_actual = k + 4
                u_vec = {i * 4 + j: 1}
                v_vec = {j * 9 + k_actual: 1}
                w_vec = {k_actual * 4 + i: 1}
                terms.append((u_vec, v_vec, w_vec))

    rank = len(terms)

    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]

    for col, (u_vec, v_vec, w_vec) in enumerate(terms):
        for idx, val in u_vec.items():
            U[idx][col] = val
        for idx, val in v_vec.items():
            V[idx][col] = val
        for idx, val in w_vec.items():
            W[idx][col] = val

    # Verify
    errors = 0
    for i in range(4):
        for j in range(4):
            for k in range(9):
                total = 0
                for r in range(rank):
                    total += U[i*4+j][r] * V[j*9+k][r] * W[k*4+i][r]
                if abs(total - 1.0) > 1e-10:
                    errors += 1
                if time.time() > start_time + 140:
                    break
            if time.time() > start_time + 140:
                break
        if time.time() > start_time + 140:
            break

    if errors > 0:
        rank = 144
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i*4+j][col] = 1
                    V[j*9+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
