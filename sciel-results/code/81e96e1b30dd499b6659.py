import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS

    # First, try Laderman's 23-term algorithm for 3x3 blocks (rank 184)
    block_size = 3
    n_blocks = n // block_size  # 2
    rank_laderman = 23 * 8  # 23 per (ib, jb, kb) triple, 2^3 = 8 triples

    U = np.zeros((n * m, rank_laderman), dtype=int)
    V = np.zeros((m * p, rank_laderman), dtype=int)
    W = np.zeros((p * n, rank_laderman), dtype=int)

    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Laderman's 23 terms: each term is (u_indices, v_indices, w_indices_with_coeffs)
    # u_indices: list of (r, c) in A block
    # v_indices: list of (r, c) in B block
    # w_indices: list of ((r, c), coeff) in C block

    laderman_terms = [
        # M1: A00*B00 -> C00
        ([(0, 0)], [(0, 0)], [((0, 0), 1)]),
        # M2: A11*B11 -> C11
        ([(1, 1)], [(1, 1)], [((1, 1), 1)]),
        # M3: A22*B22 -> C22
        ([(2, 2)], [(2, 2)], [((2, 2), 1)]),
        # M4: (A01+A02)*B10 -> C01
        ([(0, 1), (0, 2)], [(1, 0)], [((0, 1), 1)]),
        # M5: A02*(B20-B10) -> C02
        ([(0, 2)], [(2, 0), (1, 0)], [((0, 2), 1)]),
        # M6: (A10+A20)*B01 -> C10
        ([(1, 0), (2, 0)], [(0, 1)], [((1, 0), 1)]),
        # M7: A20*(B02-B01) -> C12
        ([(2, 0)], [(0, 2), (0, 1)], [((1, 2), 1)]),
        # M8: (A00-A11)*(B01+B02) -> C00(-1), C01(+1)
        ([(0, 0), (1, 1)], [(0, 1), (0, 2)], [((0, 0), -1), ((0, 1), 1)]),
        # M9: (A00-A22)*(B10+B12) -> C00(+1), C02(-1)
        ([(0, 0), (2, 2)], [(1, 0), (1, 2)], [((0, 0), 1), ((0, 2), -1)]),
        # M10: (A10-A11)*(B02+B12) -> C10(-1), C11(+1)
        ([(1, 0), (1, 1)], [(0, 2), (1, 2)], [((1, 0), -1), ((1, 1), 1)]),
        # M11: (A20-A22)*(B01+B11) -> C11(+1), C21(+1)
        ([(2, 0), (2, 2)], [(0, 1), (1, 1)], [((1, 1), 1), ((2, 1), 1)]),
        # M12: (A01-A21)*(B12+B22) -> C21(+1), C22(+1)
        ([(0, 1), (2, 1)], [(1, 2), (2, 2)], [((2, 1), 1), ((2, 2), 1)]),
        # M13: (A12-A11)*(B20+B22) -> C22(+1)
        ([(1, 2), (1, 1)], [(2, 0), (2, 2)], [((2, 2), 1)]),
        # M14: (A00-A01)*(B20+B10+B11+B12+B21+B22) -> C00(-1), C01(+1), C11(-1)
        ([(0, 0), (0, 1)], [(2, 0), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)], [((0, 0), -1), ((0, 1), 1), ((1, 1), -1)]),
        # M15: (A11-A12)*(B00+B01+B02+B10+B12+B20) -> C02(+1), C10(+1), C22(-1)
        ([(1, 1), (1, 2)], [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0)], [((0, 2), 1), ((1, 0), 1), ((2, 2), -1)]),
        # M16: (A22-A20)*(B00+B01+B02+B10+B11+B20+B21) -> C12(+1), C20(+1), C21(-1)
        ([(2, 2), (2, 0)], [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)], [((1, 2), 1), ((2, 0), 1), ((2, 1), -1)]),
        # M17: (A01-A11)*(B10+B20+B21+B22) -> C00(+1), C01(-1), C11(+1)
        ([(0, 1), (1, 1)], [(1, 0), (2, 0), (2, 1), (2, 2)], [((0, 0), 1), ((0, 1), -1), ((1, 1), 1)]),
        # M18: (A12-A22)*(B00+B01+B02+B12) -> C00(-1), C11(-1), C12(-1), C22(+1)
        ([(1, 2), (2, 2)], [(0, 0), (0, 1), (0, 2), (1, 2)], [((0, 0), -1), ((1, 1), -1), ((1, 2), -1), ((2, 2), 1)]),
        # M19: (A02-A20)*(B00+B10+B11+B21) -> C00(-1), C02(+1), C20(-1)
        ([(0, 2), (2, 0)], [(0, 0), (1, 0), (1, 1), (2, 1)], [((0, 0), -1), ((0, 2), 1), ((2, 0), -1)]),
        # M20: (A10-A00)*(B02+B11+B22) -> C10(-1), C11(+1)
        ([(1, 0), (0, 0)], [(0, 2), (1, 1), (2, 2)], [((1, 0), -1), ((1, 1), 1)]),
        # M21: (A21-A11)*(B00+B12+B22) -> C11(-1), C12(+1), C22(-1)
        ([(2, 1), (1, 1)], [(0, 0), (1, 2), (2, 2)], [((1, 1), -1), ((1, 2), 1), ((2, 2), -1)]),
        # M22: (A02-A22)*(B01+B10+B20) -> C02(-1), C21(+1)
        ([(0, 2), (2, 2)], [(0, 1), (1, 0), (2, 0)], [((0, 2), -1), ((2, 1), 1)]),
        # M23: (A10-A00)*(B01+B11+B21) -> C22(+1)
        ([(1, 0), (0, 0)], [(0, 1), (1, 1), (2, 1)], [((2, 2), 1)]),
    ]

    col = 0
    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                for u_inds, v_inds, w_inds in laderman_terms:
                    for (r, c) in u_inds:
                        U[u_idx(ib, kb, r, c), col] += 1
                    for (r, c) in v_inds:
                        V[v_idx(kb, jb, r, c), col] += 1
                    for (rc, coeff) in w_inds:
                        r, c = rc
                        W[w_idx(ib, jb, r, c), col] += coeff
                    col += 1

    # Verify Laderman decomposition
    if time.time() < deadline - 10:
        T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_true = np.zeros((n*m, m*p, p*n), dtype=int)
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    T_true[a, b, c] = 1

        if np.array_equal(T_computed, T_true):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to Strassen 2x2 blocks (rank 189)
    block_size = 2
    n_blocks = n // block_size  # 3
    rank = n_blocks ** 3 * 7  # 27 * 7 = 189

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def u_idx2(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx2(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx2(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    col = 0
    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                # M1 = (A00+A11)*(B00+B11)
                U[u_idx2(ib, kb, 0, 0), col] += 1
                U[u_idx2(ib, kb, 1, 1), col] += 1
                V[v_idx2(kb, jb, 0, 0), col] += 1
                V[v_idx2(kb, jb, 1, 1), col] += 1
                W[w_idx2(ib, jb, 0, 0), col] += 1
                W[w_idx2(ib, jb, 1, 1), col] += 1
                col += 1

                # M2 = (A10+A11)*B00
                U[u_idx2(ib, kb, 1, 0), col] += 1
                U[u_idx2(ib, kb, 1, 1), col] += 1
                V[v_idx2(kb, jb, 0, 0), col] += 1
                W[w_idx2(ib, jb, 1, 0), col] += 1
                W[w_idx2(ib, jb, 1, 1), col] -= 1
                col += 1

                # M3 = A00*(B01-B11)
                U[u_idx2(ib, kb, 0, 0), col] += 1
                V[v_idx2(kb, jb, 0, 1), col] += 1
                V[v_idx2(kb, jb, 1, 1), col] -= 1
                W[w_idx2(ib, jb, 0, 1), col] += 1
                W[w_idx2(ib, jb, 1, 1), col] += 1
                col += 1

                # M4 = A11*(B10-B00)
                U[u_idx2(ib, kb, 1, 1), col] += 1
                V[v_idx2(kb, jb, 1, 0), col] += 1
                V[v_idx2(kb, jb, 0, 0), col] -= 1
                W[w_idx2(ib, jb, 0, 0), col] += 1
                W[w_idx2(ib, jb, 1, 0), col] += 1
                col += 1

                # M5 = (A00+A01)*B11
                U[u_idx2(ib, kb, 0, 0), col] += 1
                U[u_idx2(ib, kb, 0, 1), col] += 1
                V[v_idx2(kb, jb, 1, 1), col] += 1
                W[w_idx2(ib, jb, 0, 0), col] -= 1
                W[w_idx2(ib, jb, 0, 1), col] += 1
                col += 1

                # M6 = (A10-A00)*(B00+B01)
                U[u_idx2(ib, kb, 1, 0), col] += 1
                U[u_idx2(ib, kb, 0, 0), col] -= 1
                V[v_idx2(kb, jb, 0, 0), col] += 1
                V[v_idx2(kb, jb, 0, 1), col] += 1
                W[w_idx2(ib, jb, 1, 1), col] += 1
                col += 1

                # M7 = (A01-A11)*(B10+B11)
                U[u_idx2(ib, kb, 0, 1), col] += 1
                U[u_idx2(ib, kb, 1, 1), col] -= 1
                V[v_idx2(kb, jb, 1, 0), col] += 1
                V[v_idx2(kb, jb, 1, 1), col] += 1
                W[w_idx2(ib, jb, 0, 0), col] += 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
