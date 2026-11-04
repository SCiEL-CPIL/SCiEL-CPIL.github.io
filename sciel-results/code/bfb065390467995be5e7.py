import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_strassen_56():
    """Build the rank-56 Strassen decomposition for 2x4 * 4x8."""
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * m + j, col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j * p + k, col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k * n + i, col_idx] += val

    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            if _time_left() < 5.0:
                break

            # Strassen 7 terms for 2x2x2 block
            # M1 = (A00+A11)(B00+B11) -> C00:1, C11:1
            add_term(
                {(0, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1

            # M2 = (A10+A11)B00 -> C10:1, C11:-1
            add_term(
                {(1, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1},
                {(k0, 1): 1, (k0+1, 1): -1},
                col
            )
            col += 1

            # M3 = A00(B01-B11) -> C01:1, C11:1
            add_term(
                {(0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(k0+1, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1

            # M4 = A11(B10-B00) -> C00:1, C10:1
            add_term(
                {(1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(k0, 0): 1, (k0, 1): 1},
                col
            )
            col += 1

            # M5 = (A00+A01)B11 -> C00:-1, C01:1
            add_term(
                {(0, j0): 1, (0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(k0, 0): -1, (k0+1, 0): 1},
                col
            )
            col += 1

            # M6 = (A10-A00)(B00+B01) -> C11:1
            add_term(
                {(1, j0): 1, (0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(k0+1, 1): 1},
                col
            )
            col += 1

            # M7 = (A01-A11)(B10+B11) -> C00:1
            add_term(
                {(0, j0+1): 1, (1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1},
                col
            )
            col += 1

    return U, V, W

def construct():
    n, m, p = 2, 4, 8

    U, V, W = _build_strassen_56()

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
