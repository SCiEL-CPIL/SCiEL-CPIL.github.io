import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    _start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 28

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen 2x2x2 terms: (u_local, v_local, w_local)
    # local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_off + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_off + r) * p + (b_col_off + c)
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_off + c) * n + r
                W[g_idx, col] += coeff

    # Four 2x2x2 blocks
    apply_strassen(0, 0, 0, 0, 0)   # A1*B11 -> C11
    apply_strassen(2, 2, 0, 0, 7)   # A2*B21 -> C21
    apply_strassen(0, 0, 2, 2, 14)  # A1*B12 -> C12
    apply_strassen(2, 2, 2, 2, 21)  # A2*B22 -> C22

    # Verify correctness with a quick test
    if time.time() - _start_time < 130:
        Uf = U.astype(float)
        Vf = V.astype(float)
        Wf = W.astype(float)
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += Uf[ii * m + jj, cc] * Vf[jj * p + kk, cc] * Wf[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-10):
            # Fallback: naive rank-32 decomposition
            rank_naive = n * m * p
            U_n = np.zeros((n * m, rank_naive), dtype=int)
            V_n = np.zeros((m * p, rank_naive), dtype=int)
            W_n = np.zeros((p * n, rank_naive), dtype=int)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_n[i * m + j, idx] = 1
                        V_n[j * p + k, idx] = 1
                        W_n[k * n + i, idx] = 1
                        idx += 1
            return {"U": U_n.tolist(), "V": V_n.tolist(), "W": W_n.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B

    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc

    if not np.allclose(C, Cf):
        print("Mismatch")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
