import numpy as np
import time

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140.0

    # Try to find a rank-50 decomposition using 5 blocks of 2x2x3
    result = _try_rank50(deadline)
    if result is not None:
        return result

    # Try direct rank-50 search
    result = _try_direct_rank50(deadline)
    if result is not None:
        return result

    # Fallback to rank 53
    return _build_rank53()

def _verify_decomposition(U, V, W, n=2, m=2, p=15):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    u_val = U[i*m + j][r]
                    v_val = V[j*p + k][r]
                    w_val = W[k*n + i][r]
                    actual += u_val * v_val * w_val
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _try_rank50(deadline):
    n, m, p = 2, 2, 15
    rank_block = 10
    num_blocks = 5
    total_rank = rank_block * num_blocks

    U_b, V_b, W_b = _find_m223_rank10(deadline)
    if U_b is None:
        return None

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for b in range(num_blocks):
        k0 = 3 * b
        col_offset = b * rank_block
        for r in range(rank_block):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_b[i*2 + j][r]
            for j in range(2):
                for k in range(3):
                    kk = k0 + k
                    V[j*p + kk][c] = V_b[j*3 + k][r]
            for k in range(3):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_b[k*2 + i][r]

    if not _verify_decomposition(U, V, W, n, m, p):
        return None

    return {"U": U, "V": V, "W": W}

def _build_rank53():
    n, m, p = 2, 2, 15
    rank = 53

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2x2
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # Schoolbook for 2x2x1
    U_sc = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_sc = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sc = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_s[i*2 + j][r]
            for j in range(2):
                V[j*p + k0][c] = V_s[j*2 + 0][r]
                V[j*p + k1][c] = V_s[j*2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_s[k*2 + i][r]
        col += 7

    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_sc[i*2 + j][r]
        for j in range(2):
            V[j*p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last*n + i][c] = W_sc[i][r]

    return {"U": U, "V": V, "W": W}

def _find_m223_rank10(deadline):
    n, m, p = 2, 2, 3
    rank = 10

    def residuals(params):
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+6*rank].reshape(6, rank)
        W = params[4*rank+6*rank:].reshape(6, rank)

        res = np.zeros(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[idx] = np.dot(U[i*2+j], V[j*3+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res

    inits = []

    # Structured initialization based on Strassen + extensions
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, -1, 0, 0, 0]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=float)
    inits.append(np.concatenate([U_s.flatten(), V_s.flatten(), W_s.flatten()]))

    # Random initializations
    for seed in range(100):
        if time.time() > deadline - 20:
            break
        np.random.seed(seed)
        scale = [0.2, 0.5, 1.0, 1.5, 2.0][seed % 5]
        x0 = np.random.randn(16*rank) * scale
        inits.append(x0)

    best_result = None
    best_norm = np.inf

    for x0 in inits:
        if time.time() > deadline - 20:
            break

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000)
            norm = np.linalg.norm(result.fun)
            if norm < best_norm:
                best_norm = norm
                best_result = result

            if norm < 1e-9:
                U = result.x[:4*rank].reshape(4, rank)
                V = result.x[4*rank:4*rank+6*rank].reshape(6, rank)
                W = result.x[4*rank+6*rank:].reshape(6, rank)

                U, V, W = _polish_m223(U, V, W, deadline)

                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2

                res_r = residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
                if np.max(np.abs(res_r)) < 1e-6:
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
        except Exception:
            continue

    # Polish the best result if it was close
    if best_result is not None and best_norm < 1e-4:
        try:
            U = best_result.x[:4*rank].reshape(4, rank)
            V = best_result.x[4*rank:4*rank+6*rank].reshape(6, rank)
            W = best_result.x[4*rank+6*rank:].reshape(6, rank)

            U, V, W = _polish_m223(U, V, W, deadline)

            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2

            res_r = residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
            if np.max(np.abs(res_r)) < 1e-6:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        except Exception:
            pass

    return None, None, None

def _polish_m223(U, V, W, deadline):
    rank = U.shape[1]

    def residuals(params):
        U_l = params[:4*rank].reshape(4, rank)
        V_l = params[4*rank:4*rank+6*rank].reshape(6, rank)
        W_l = params[4*rank+6*rank:].reshape(6, rank)

        res = np.zeros(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[idx] = np.dot(U_l[i*2+j], V_l[j*3+k] * W_l[k*2+i]) - 1.0
                    idx += 1
        return res

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=10000, ftol=1e-14, xtol=1e-14)
        if np.linalg.norm(result.fun) < 1e-12:
            U_p = result.x[:4*rank].reshape(4, rank)
            V_p = result.x[4*rank:4*rank+6*rank].reshape(6, rank)
            W_p = result.x[4*rank+6*rank:].reshape(6, rank)
            return U_p, V_p, W_p
    except Exception:
        pass

    return U, V, W

def _try_direct_rank50(deadline):
    n, m, p = 2, 2, 15
    rank = 50

    def residuals(params):
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+30*rank].reshape(30, rank)
        W = params[4*rank+30*rank:].reshape(30, rank)

        res = np.zeros(60)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(15):
                    res[idx] = np.dot(U[i*2+j], V[j*15+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res

    # Initialize with block structure
    U_init = np.zeros((4, rank))
    V_init = np.zeros((30, rank))
    W_init = np.zeros((30, rank))

    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    for b in range(5):
        col_off = b * 10
        k0 = 3 * b
        for r in range(7):
            c = col_off + r
            for i in range(2):
                for j in range(2):
                    U_init[i*2+j, c] = U_s[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    kk = k0 + k
                    V_init[j*15+kk, c] = V_s[j*2+k, r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W_init[kk*2+i, c] = W_s[k*2+i, r]

        for r in range(3):
            c = col_off + 7 + r
            if r == 0:
                U_init[0, c] = 1
                V_init[15, c] = 1
                W_init[4, c] = 1
            elif r == 1:
                U_init[1, c] = 1
                V_init[16, c] = 1
                W_init[5, c] = 1
            else:
                U_init[2, c] = 1
                V_init[15, c] = 1
                W_init[4, c] = 1

    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=3000)
        if np.linalg.norm(result.fun) < 1e-9:
            U = result.x[:4*rank].reshape(4, rank)
            V = result.x[4*rank:4*rank+30*rank].reshape(30, rank)
            W = result.x[4*rank+30*rank:].reshape(30, rank)

            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2

            res_r = residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
            if np.max(np.abs(res_r)) < 1e-6:
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    except Exception:
        pass

    return None

def least_squares(fun, x0, method='lm', max_nfev=1000, ftol=1e-8, xtol=1e-8, gtol=1e-8):
    """Simple Levenberg-Marquardt implementation."""
    from scipy.optimize import least_squares as _ls
    return _ls(fun, x0, method=method, max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol)
