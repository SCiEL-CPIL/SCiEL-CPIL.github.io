import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def build_block_strassen():
        rank = 56
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)

        for u in range(2):
            for v in range(4):
                base_col = 7 * (4 * u + v)
                
                a11 = 0 * m + 2 * u
                a12 = 0 * m + 2 * u + 1
                a21 = 1 * m + 2 * u
                a22 = 1 * m + 2 * u + 1
                a_local_to_global = {0: a11, 1: a12, 2: a21, 3: a22}

                b11 = (2 * u) * p + 2 * v
                b12 = (2 * u) * p + 2 * v + 1
                b21 = (2 * u + 1) * p + 2 * v
                b22 = (2 * u + 1) * p + 2 * v + 1
                b_local_to_global = {0: b11, 1: b12, 2: b21, 3: b22}

                c11 = (2 * v) * n + 0
                c12 = (2 * v + 1) * n + 0
                c21 = (2 * v) * n + 1
                c22 = (2 * v + 1) * n + 1
                c_local_to_global = {0: c11, 1: c12, 2: c21, 3: c22}

                for t in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                    col = base_col + t

                    for local_idx, coeff in u_coeffs:
                        U[a_local_to_global[local_idx], col] += coeff
                    for local_idx, coeff in v_coeffs:
                        V[b_local_to_global[local_idx], col] += coeff
                    for local_idx, coeff in w_coeffs:
                        W[c_local_to_global[local_idx], col] += coeff

        return U, V, W

    def verify(U, V, W, n, m, p, atol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True

    U, V, W = build_block_strassen()

    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
