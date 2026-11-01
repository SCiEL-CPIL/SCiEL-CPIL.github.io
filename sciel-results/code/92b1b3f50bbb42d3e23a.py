"""Rank-minimal decomposition for 2x2 by 2x16 matrix multiplication.

Strategy: view the 2x16 product as 8 independent 2x2 products (columns
grouped in pairs). Each 2x2 product is computed with a rank-7 Strassen
decomposition, giving a total rank of 8*7 = 56 (vs schoolbook 64).
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 150.0 - 5.0  # safety margin

    n, m, p = DIMENSIONS  # 2, 2, 16

    # Standard Strassen rank-7 decomposition for 2x2.
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A21)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # A rows: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B rows: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C rows (trace convention k*n+i): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3

    A_factors = [
        [1, 1],  # M1
        [0, 0],  # M2
        [1, 0],  # M3
        [0, 0],  # M4
        [1, 1],  # M5
        [1, 0],  # M6
        [0, 1],  # M7
    ]
    A_rows = [
        [0, 3],  # M1: A11+A22
        [2, 3],  # M2: A21+A22
        [0],     # M3: A11
        [3],     # M4: A22
        [0, 1],  # M5: A11+A12
        [2, 0],  # M6: A21-A11
        [1, 2],  # M7: A12-A21
    ]
    A_signs = [
        [1, 1],
        [1, 1],
        [1],
        [1],
        [1, 1],
        [1, -1],
        [1, -1],
    ]
    B_factors = [
        [1, 1],  # M1
        [1],      # M2
        [1, 0],   # M3
        [0, 1],   # M4
        [0, 1],   # M5
        [1, 1],   # M6
        [1, 1],   # M7
    ]
    B_rows = [
        [0, 3],  # M1: B11+B22
        [0],     # M2: B11
        [1, 3],  # M3: B12-B22
        [2, 0],  # M4: B21-B11
        [3],     # M5: B22
        [0, 1],  # M6: B11+B12
        [2, 3],  # M7: B21+B22
    ]
    B_signs = [
        [1, 1],
        [1],
        [1, -1],
        [1, -1],
        [1],
        [1, 1],
        [1, 1],
    ]
    C_rows = [
        [0, 3, 1, 2],  # M1
        [3, 2],         # M2
        [1, 3],         # M3
        [0, 2],         # M4
        [0, 1],         # M5
        [3],            # M6
        [0],            # M7
    ]
    C_signs = [
        [1, 1, -1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1],
        [1],
    ]

    rank = 8 * 7
    U = [[0] * rank for _ in range(n * m)]   # 4 x rank
    V = [[0] * rank for _ in range(m * p)]   # 32 x rank
    W = [[0] * rank for _ in range(p * n)]   # 32 x rank

    for g in range(8):
        for s in range(7):
            col = g * 7 + s
            # U
            for idx, row in enumerate(A_rows[s]):
                U[row][col] += A_signs[s][idx]
            # V
            for idx, row in enumerate(B_rows[s]):
                V[g * 4 + row][col] += B_signs[s][idx]
            # W
            for idx, row in enumerate(C_rows[s]):
                W[g * 4 + row][col] += C_signs[s][idx]

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Verify correctness
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
        if not ok:
            break

    if not ok:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
