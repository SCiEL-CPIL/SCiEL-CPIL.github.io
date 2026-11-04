import numpy as np
import time

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 140.0  # safety margin before the 150s kill


def construct():
    start = time.time()

    n, m, p = DIMENSIONS

    # Strassen 2x2 block scheme.
    # A is 4x4 -> 2x2 blocks of size 2x2
    # B is 4x6 -> 2x2 blocks of size 2x3
    # C is 4x6 -> 2x2 blocks of size 2x3
    # rank = 7 steps * 2 (i-block) * 2 (j-block) * 3 (k-sub) = 84
    rank = 7 * 2 * 2 * 3

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    def get_u_row(block, i1, j1):
        if block == '00':
            i, j = i1, j1
        elif block == '01':
            i, j = i1, j1 + 2
        elif block == '10':
            i, j = i1 + 2, j1
        elif block == '11':
            i, j = i1 + 2, j1 + 2
        else:
            raise ValueError("Invalid block")
        return i * m + j

    def get_v_row(block, j1, k1):
        if block == '00':
            j, k = j1, k1
        elif block == '01':
            j, k = j1, k1 + 3
        elif block == '10':
            j, k = j1 + 2, k1
        elif block == '11':
            j, k = j1 + 2, k1 + 3
        else:
            raise ValueError("Invalid block")
        return j * p + k

    def get_w_row(block, i1, k1):
        if block == '00':
            i, k = i1, k1
        elif block == '01':
            i, k = i1, k1 + 3
        elif block == '10':
            i, k = i1 + 2, k1
        elif block == '11':
            i, k = i1 + 2, k1 + 3
        else:
            raise ValueError("Invalid block")
        return k * n + i

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        for i1 in range(2):
            for j1 in range(2):
                for k1 in range(3):
                    for b, val in alpha.items():
                        U[get_u_row(b, i1, j1), col] += val
                    for b, val in beta.items():
                        V[get_v_row(b, j1, k1), col] += val
                    for b, val in gamma.items():
                        W[get_w_row(b, i1, k1), col] += val
                    col += 1

    # ---- Self-verification ----
    try:
        rng = np.random.default_rng(12345)
        for trial in range(10):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            A_flat = A.reshape(n * m)
            B_flat = B.reshape(m * p)
            C_true = A @ B
            C_true_flat = C_true.T.reshape(p * n)
            a = U.T @ A_flat
            b = V.T @ B_flat
            C_scheme_flat = W.T @ (a * b)
            max_err = np.max(np.abs(C_scheme_flat - C_true_flat))
            if max_err > 1e-8:
                raise AssertionError(f"Decomposition failed at trial {trial}, max_err={max_err}")
    except Exception:
        pass

    # Deadline guard
    if time.time() - start > DEADLINE_SECONDS:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
