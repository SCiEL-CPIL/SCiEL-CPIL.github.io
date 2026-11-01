import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 115.0  # safety margin before 120s kill

    n, m, p = DIMENSIONS
    rank = 28

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen 2x2x2 terms: (u_local, v_local, w_local)
    # Each local index 0..3 corresponds to 2x2 matrix entry
    # Using standard Strassen decomposition
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def map_u_a(a_side, local_idx):
        # A is 2x4. Left half (cols 0-1) or right half (cols 2-3)
        # local_idx 0..3 maps to 2x2 subblock
        r = local_idx // 2
        c = local_idx % 2
        if a_side == 'L':
            return r * 4 + c
        else:
            return r * 4 + 2 + c

    def map_v_b(b_block, local_idx):
        # B is 4x4. Block: TL, TR, BL, BR (each 2x2)
        r = local_idx // 2
        c = local_idx % 2
        row_off = {'TL': 0, 'TR': 0, 'BL': 2, 'BR': 2}[b_block]
        col_off = {'TL': 0, 'TR': 2, 'BL': 0, 'BR': 2}[b_block]
        global_r = row_off + r
        global_c = col_off + c
        return global_r * 4 + global_c

    def map_w_c(c_side, local_idx):
        # C is 2x4. Left (cols 0-1) or right (cols 2-3)
        r = local_idx // 2
        c = local_idx % 2
        if c_side == 'L':
            return c * 2 + r  # W[k*n+i, r] -> for C[r,c]: index = c*n + r
        else:
            return (2 + c) * 2 + r

    col = 0
    for a_side, b_block, c_side in [('L', 'TL', 'L'), ('R', 'BL', 'L'),
                                    ('L', 'TR', 'R'), ('R', 'BR', 'R')]:
        for u_local, v_local, w_local in strassen_terms:
            for idx, coeff in u_local:
                U[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_local:
                V[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_local:
                W[map_w_c(c_side, idx), col] += coeff
            col += 1

        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
