import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _strassen_44():
    n, m, p = DIMENSIONS
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    # Part 1: 16 terms for row 0 of A
    for k in range(4):
        for j in range(4):
            U[j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3][col] = 1
            col += 1
    # Part 2: 28 terms for rows 1-2 of A using Strassen on 2x2 blocks
    strassen_terms = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2),
    ]
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in strassen_terms:
            for (xi, yi), ux in x_coeffs.items():
                U[(ar + xi) * 4 + (ac + yi)][col] += ux
            for (yi, xi), vy in y_coeffs.items():
                V[(br + yi) * 4 + (bc + xi)][col] += vy
            for (zi, zj), wz in z_coeffs.items():
                W[(cc + zj) * 3 + (cr + zi)][col] += wz
            col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140
    result = _strassen_44()
    n, m, p = DIMENSIONS
    err = _verify(result["U"], result["V"], result["W"], n, m, p)
    if err > 1e-9:
        total_rank = n * m * p
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
    return result
