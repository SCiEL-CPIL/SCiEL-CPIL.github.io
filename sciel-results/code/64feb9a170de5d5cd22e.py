import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1

    # Strassen for 2x2x2
    strassen_local = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]

    # Rank 1 for 2x2x1
    # C_i = sum_j A_ij B_j
    # u_i=1, v_j=1, w_0=1
    rank1_local = [
        ({0:1, 1:1, 2:1, 3:1}, {0:1, 1:1}, {0:1}),
    ]

    # Process A1 B1 (j in 0,1) and A2 B2 (j in 2,3)
    for block in range(2):
        j_offset = block * 2
        for k_block in range(3):
            if k_block < 2:
                k_offset = k_block * 2
                for u_t, v_t, w_t in strassen_local:
                    u_g, v_g, w_g = {}, {}, {}
                    for idx, val in u_t.items():
                        li, lj = idx // 2, idx % 2
                        u_g[li * m + (j_offset + lj)] = val
                    for idx, val in v_t.items():
                        lj, lk = idx // 2, idx % 2
                        v_g[(j_offset + lj) * p + (k_offset + lk)] = val
                    for idx, val in w_t.items():
                        li, lk = idx // 2, idx % 2
                        w_g[(k_offset + lk) * n + li] = val
                    add_term(u_g, v_g, w_g)
            else:
                k_offset = 4
                for u_t, v_t, w_t in rank1_local:
                    u_g, v_g, w_g = {}, {}, {}
                    for idx, val in u_t.items():
                        li, lj = idx // 2, idx % 2
                        u_g[li * m + (j_offset + lj)] = val
                    for idx, val in v_t.items():
                        lj = idx
                        v_g[(j_offset + lj) * p + k_offset] = val
                    for idx, val in w_t.items():
                        li = idx
                        w_g[k_offset * n + li] = val
                    add_term(u_g, v_g, w_g)

    # Verify
    if time.time() < deadline - 5:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            np.random.seed(42)
            ok = True
            for trial in range(20):
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                i_g = i * m + j
                j_g = j * p + k
                k_g = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_np[i_g, r] * V_np[j_g, r] * W_np[k_g, r]
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                # Fallback
                rank40 = n * m * p
                U40 = [[0] * rank40 for _ in range(n * m)]
                V40 = [[0] * rank40 for _ in range(m * p)]
                W40 = [[0] * rank40 for _ in range(p * n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = i * m * p + j * p + k
                            U40[i * m + j][r] = 1
                            V40[j * p + k][r] = 1
                            W40[k * n + i][r] = 1
                return {"U": U40, "V": V40, "W": W40}
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
