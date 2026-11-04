import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin

    n, m, p = 3, 3, 3
    size_u = n * m  # 9
    size_v = m * p  # 9
    size_w = p * n  # 9

    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Strassen-Winograd Rank 23 Decomposition
    # U: 9x23, V: 9x23, W: 9x23
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)

    # Data structure for the 23 terms: (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    # This compact format is less prone to copy-paste errors than 23 blocks of assignments.
    terms = [
        # Term 0
        ([0, 4, 8], [1, -1, -1], [0, 4, 8], [1, 1, 1], [0], [1]),
        # Term 1
        ([4, 8], [1, -1], [4, 8], [1, 1], [0, 4, 8], [1, 1, 1]),
        # Term 2
        ([8], [1], [8], [1], [0, 4, 8], [1, 1, 1]),
        # Term 3
        ([1, 5], [1, -1], [3, 6], [1, -1], [3, 4, 7], [1, -1, -1]),
        # Term 4
        ([2, 3], [1, -1], [6, 0], [1, -1], [6, 3, 0], [1, -1, -1]),
        # Term 5
        ([3, 6], [1, -1], [0, 1], [1, -1], [3, 4, 7], [1, -1, -1]),
        # Term 6
        ([1, 2], [1, -1], [2, 0], [1, -1], [2, 1, 0], [1, -1, -1]),
        # Term 7
        ([5, 8], [1, -1], [6, 7], [1, -1], [6, 8, 7], [1, -1, -1]),
        # Term 8
        ([6, 7], [1, -1], [4, 5], [1, -1], [6, 7, 5], [1, -1, -1]),
        # Term 9
        ([1, 4, 7], [1, 1, 1], [2, 5, 8], [1, -1, -1], [2, 4, 7], [1, -1, -1]),
        # Term 10
        ([4, 7], [1, 1], [5, 8], [1, 1], [5, 7, 8], [1, 1, 1]),
        # Term 11
        ([7], [1], [8], [1], [8, 7], [1, 1]),
        # Term 12
        ([2, 5, 8], [1, 1, 1], [1, 4, 7], [1, -1, -1], [2, 5, 8], [1, -1, -1]),
        # Term 13
        ([5, 8], [1, 1], [4, 7], [1, 1], [5, 8, 7], [1, 1, 1]),
        # Term 14
        ([8], [1], [7], [1], [8, 7], [1, 1]),
        # Term 15
        ([0, 3, 6], [1, 1, 1], [2, 5, 8], [1, -1, -1], [2, 3, 6], [1, -1, -1]),
        # Term 16
        ([3, 6], [1, 1], [5, 8], [1, 1], [3, 6, 8], [1, 1, 1]),
        # Term 17
        ([6], [1], [8], [1], [6, 8], [1, 1]),
        # Term 18
        ([0, 3, 6], [1, 1, 1], [1, 4, 7], [1, -1, -1], [1, 3, 6], [1, -1, -1]),
        # Term 19
        ([3, 6], [1, 1], [4, 7], [1, 1], [3, 6, 7], [1, 1, 1]),
        # Term 20
        ([6], [1], [7], [1], [6, 7], [1, 1]),
        # Term 21
        ([0, 3, 6], [1, 1, 1], [0, 3, 6], [1, -1, -1], [0, 3, 6], [1, -1, -1]),
        # Term 22
        ([3, 6], [1, 1], [3, 6], [1, 1], [3, 6, 0], [1, 1, 1]),
    ]

    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for idx, val in zip(u_idx, u_val):
            U[idx, r] = val
        for idx, val in zip(v_idx, v_val):
            V[idx, r] = val
        for idx, val in zip(w_idx, w_val):
            W[idx, r] = val

    # Verify the decomposition
    R = np.einsum('ar,br,cr->abc', U, V, W)
    if np.allclose(R, T, atol=1e-9):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback to schoolbook if hardcoded fails
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
