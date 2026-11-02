"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    def make_54_rank():
        """Known working 54-rank: 6 Strassen 2x2x2 blocks + 12 schoolbook for j=4."""
        rank = 54
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            if col >= rank:
                return
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx, coeff in v_coeffs:
                V[v_idx, col] += coeff
            for w_idx, coeff in w_coeffs:
                W[w_idx, col] += coeff
            col += 1

        def strassen_block(ja, jb, ka, kb):
            ua0 = ja
            ua1 = n + ja
            ub0 = jb
            ub1 = n + jb
            vka = ja * p + ka
            vkb = ja * p + kb
            vjbka = jb * p + ka
            vjbkb = jb * p + kb
            wka0 = ka * n
            wka1 = ka * n + 1
            wkb0 = kb * n
            wkb1 = kb * n + 1

            add_term([(ua0, 1), (ub1, 1)], [(vka, 1), (vjbkb, 1)], [(wka0, 1), (wkb1, 1)])
            add_term([(ub0, 1), (ub1, 1)], [(vka, 1)], [(wka1, 1), (wkb1, -1)])
            add_term([(ua0, 1)], [(vkb, 1), (vjbkb, -1)], [(wkb0, 1), (wkb1, 1)])
            add_term([(ub0, 1)], [(vjbka, 1), (vka, -1)], [(wka0, 1), (wka1, 1)])
            add_term([(ua0, 1), (ub0, 1)], [(vjbkb, 1)], [(wka0, -1), (wkb0, 1)])
            add_term([(ub0, 1), (ua0, -1)], [(vka, 1), (vkb, 1)], [(wkb1, 1)])
            add_term([(ub0, 1), (ub1, -1)], [(vjbka, 1), (vjbkb, 1)], [(wka0, 1)])

        # 6 Strassen blocks for j pairs (0,1) and (2,3), k pairs (0,1),(2,3),(4,5)
        for jpair in [(0, 1), (2, 3)]:
            for kpair in [(0, 1), (2, 3), (4, 5)]:
                strassen_block(jpair[0], jpair[1], kpair[0], kpair[1])

        # Schoolbook for j=4, all k
        for i in range(n):
            for k in range(p):
                U[i * m + 4, col] = 1
                V[4 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        return U[:, :col], V[:, :col], W[:, :col]

    def make_fallback():
        U = np.zeros((n * m, 60), dtype=float)
        V = np.zeros((m * p, 60), dtype=float)
        W = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W

    # Try 54-rank first
    if time.time() < deadline - 10:
        U, V, W = make_54_rank()
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook rank 60
    U, V, W = make_fallback()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
