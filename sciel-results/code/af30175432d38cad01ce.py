import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    HARD_DEADLINE = 140.0  # safety margin before the 150s kill

    # ---- Fallback: Rank 21 Strassen (3 independent 2x2 blocks) ----
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]

        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l, val in v_coeffs.items():
                    if l == 0: v_idx = 0 * p + 2 * b
                    elif l == 1: v_idx = 0 * p + 2 * b + 1
                    elif l == 2: v_idx = 1 * p + 2 * b
                    elif l == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    # ---- Non-zero tensor entries (24 of them) ----
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    num_nonzero = len(indices)
    A_idx = np.array([a for (a, b, c) in indices])
    B_idx = np.array([b for (a, b, c) in indices])
    C_idx = np.array([c for (a, b, c) in indices])

    def verify_half_integer(U, V, W, tol=1e-9):
        """U,V,W are (rows, rank) arrays of half-integers. Check exactness."""
        max_err = 0.0
        for a, b, c in indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        return max_err < tol

    # ---- Target rank 20 search ----
    target_rank = 20
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_size = size_u + size_v + size_w

    def residual(x):
        U = x[:size_u].reshape((n * m, target_rank))
        V = x[size_u:size_u + size_v].reshape((m * p, target_rank))
        W = x[size_u + size_v:].reshape((p * n, target_rank))
        # For each entry e: sum_r U[a_e, r] V[b_e, r] W[c_e, r]
        res = np.einsum('ar,br,cr->e', U[A_idx], V[B_idx], W[C_idx])
        return res - 1.0

    def try_polish(x0, max_nfev):
        try:
            res = opt.least_squares(residual, x0, method='trf',
                                    max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            return res
        except Exception:
            return None

    def snap_and_check(res):
        U_opt = res.x[:size_u].reshape((n * m, target_rank))
        V_opt = res.x[size_u:size_u + size_v].reshape((m * p, target_rank))
        W_opt = res.x[size_u + size_v:].reshape((p * n, target_rank))
        U_int = np.round(U_opt * 2) / 2
        V_int = np.round(V_opt * 2) / 2
        W_int = np.round(W_opt * 2) / 2
        if verify_half_integer(U_int, V_int, W_int):
            def clean(arr):
                out = arr.tolist()
                for i in range(len(out)):
                    for j in range(len(out[i])):
                        if out[i][j] == 0:
                            out[i][j] = 0
                return out
            return {"U": clean(U_int), "V": clean(V_int), "W": clean(W_int)}
        return None

    # ---- Build initial guesses ----
    np.random.seed(42)
    x0_candidates = []

    # (a) Pure random restarts
    for _ in range(6):
        x0_candidates.append(np.random.normal(0, 0.5, total_size))

    # (b) Initialize from the rank-21 Strassen solution, then collapse to rank 20.
    #     This exploits the block structure: 3 blocks x 7 products = 21, we drop one.
    def build_strassen_21_flat():
        U = np.zeros((n * m, 21))
        V = np.zeros((m * p, 21))
        W = np.zeros((p * n, 21))
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx, r] = val
                for l, val in v_coeffs.items():
                    if l == 0: v_idx = 0 * p + 2 * b
                    elif l == 1: v_idx = 0 * p + 2 * b + 1
                    elif l == 2: v_idx = 1 * p + 2 * b
                    elif l == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx, r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    W[k_global * n + i, r] = val
                r += 1
        return U, V, W

    U21, V21, W21 = build_strassen_21_flat()
    # Collapse to rank 20 by dropping the last column (one redundant product) and adding small noise
    for drop in range(21):
        U20 = np.delete(U21, drop, axis=1)
        V20 = np.delete(V21, drop, axis=1)
        W20 = np.delete(W21, drop, axis=1)
        # small perturbation to break degeneracy
        U20 = U20 + np.random.normal(0, 0.1, U20.shape)
        V20 = V20 + np.random.normal(0, 0.1, V20.shape)
        W20 = W20 + np.random.normal(0, 0.1, W20.shape)
        x0 = np.concatenate([U20.ravel(), V20.ravel(), W20.ravel()])
        x0_candidates.append(x0)

    # ---- Run optimization over candidates with deadline guard ----
    best_result = None
    for x0 in x0_candidates:
        if time.time() - start_time > HARD_DEADLINE - 5.0:
            break
        res = try_polish(x0, max_nfev=400)
        if res is None:
            continue
        if (res.success or np.linalg.norm(res.fun) < 1e-5) and np.linalg.norm(res.fun) < 1e-4:
            # high-precision polish
            res2 = try_polish(res.x, max_nfev=600)
            if res2 is not None and np.linalg.norm(res2.fun) < np.linalg.norm(res.fun):
                res = res2
            found = snap_and_check(res)
            if found is not None:
                best_result = found
                break
        if time.time() - start_time > HARD_DEADLINE - 3.0:
            break

    if best_result is not None:
        return best_result

    return get_rank_21()
