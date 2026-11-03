import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9

    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Target tensor: T[i,j,k] = 1  (the matmul tensor under trace convention)
    ones_indices = [(i * m + j, j * p + k, k * n + i)
                    for i in range(n) for j in range(m) for k in range(p)]
    ones_set = set(ones_indices)
    all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]

    # Flat target vector
    targets = np.zeros(nm * mp * pn)
    for a, b, c in ones_indices:
        targets[a * mp * pn + b * pn + c] = 1.0

    def tensor_from(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)

    def verify(U, V, W, tol=1e-8):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        T = tensor_from(U, V, W)
        for a, b, c in ones_indices:
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        for a, b, c in zero_indices:
            if abs(T[a, b, c]) > tol:
                return False
        return True

    def snap_to_half_integers(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None

    def snap_to_integers(U, V, W):
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None

    # ---------------- Verified Rank-26 base (Strassen core + border) ----------------
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))

    def set_term(col, u_entries, v_entries, w_entries):
        for (i, j, c) in u_entries:
            U26[idx_u(i, j), col] = c
        for (j, k, c) in v_entries:
            V26[idx_v(j, k), col] = c
        for (k, i, c) in w_entries:
            W26[idx_w(k, i), col] = c

    # Strassen 7 terms on the 2x2 core (indices 0,1)
    set_term(0, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    set_term(1, [(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(0,1,-1)])
    set_term(2, [(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])
    set_term(3, [(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])
    set_term(4, [(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)])
    set_term(5, [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
    set_term(6, [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])

    # Border terms: one per (i,j,k) with any index == 2
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                set_term(col, [(i,j,1)], [(j,k,1)], [(k,i,1)])
                col += 1
    assert col == R26, f"Expected {R26} cols, got {col}"

    best_U, best_V, best_W = U26, V26, W26
    best_rank = R26
    if not verify(U26, V26, W26):
        # Should never happen; fall back to pure schoolbook rank 27
        R27 = 27
        U27 = np.zeros((nm, R27)); V27 = np.zeros((mp, R27)); W27 = np.zeros((pn, R27))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[idx_u(i,j), c] = 1
                    V27[idx_v(j,k), c] = 1
                    W27[idx_w(k,i), c] = 1
                    c += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}

    # ---------------- Optimization to lower the rank ----------------
    from scipy.optimize import least_squares

    def try_optimize_rank(R, time_budget, init_U=None, init_V=None, init_W=None,
                          max_restarts=20, max_nfev=4000):
        if time.time() > start_time + time_budget:
            return None
        best_cost = float('inf')
        best_x = None
        np.random.seed(42)
        for r in range(max_restarts):
            if time.time() > start_time + time_budget:
                break
            if r == 0 and init_U is not None:
                x0 = np.concatenate([
                    init_U.flatten(), init_V.flatten(), init_W.flatten()])
            else:
                x0 = np.concatenate([
                    np.random.randn(nm, R) * 0.5,
                    np.random.randn(mp, R) * 0.5,
                    np.random.randn(pn, R) * 0.5]).flatten()

            def residual(x):
                U_f = x[:nm * R].reshape(nm, R)
                V_f = x[nm * R:2 * nm * R].reshape(mp, R)
                W_f = x[2 * nm * R:3 * nm * R].reshape(pn, R)
                T = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
                return T.flatten() - targets

            try:
                sol = least_squares(residual, x0, method='trf',
                                    max_nfev=max_nfev, ftol=1e-14,
                                    xtol=1e-14, gtol=1e-14)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_x = sol.x
                    if best_cost < 1e-13:
                        break
            except Exception:
                pass

        if best_x is None or best_cost > 1e-10:
            return None
        U_sol = best_x[:nm * R].reshape(nm, R)
        V_sol = best_x[nm * R:2 * nm * R].reshape(mp, R)
        W_sol = best_x[2 * nm * R:3 * nm * R].reshape(pn, R)
        snapped = snap_to_half_integers(U_sol, V_sol, W_sol)
        if snapped is None:
            snapped = snap_to_integers(U_sol, V_sol, W_sol)
        return snapped

    # Border columns (7..25) are the mergeable candidates.
    border_cols = list(range(7, R26))

    # Attempt descending ranks 25, 24, 23.
    for target_rank in (25, 24, 23):
        if time.time() > deadline - 15.0:
            break
        remaining = deadline - 15.0 - time.time()
        if remaining <= 5.0:
            break
        found = False
        n_drop = R26 - target_rank
        # Warm-start: drop 'n_drop' border columns, then optimize the rest to rank target_rank.
        # Also try pure random restarts.
        attempts = 0
        max_attempts = 40
        for drop_set in itertools.combinations(border_cols, n_drop):
            if time.time() > deadline - 15.0:
                break
            if attempts >= max_attempts:
                break
            attempts += 1
            keep = [c for c in range(R26) if c not in drop_set]
            # Build an init of width target_rank from kept columns (pad if fewer)
            U_init = U26[:, keep]
            V_init = V26[:, keep]
            W_init = W26[:, keep]
            if U_init.shape[1] < target_rank:
                pad = target_rank - U_init.shape[1]
                U_init = np.hstack([U_init, np.zeros((nm, pad))])
                V_init = np.hstack([V_init, np.zeros((mp, pad))])
                W_init = np.hstack([W_init, np.zeros((pn, pad))])
            elif U_init.shape[1] > target_rank:
                U_init = U_init[:, :target_rank]
                V_init = V_init[:, :target_rank]
                W_init = W_init[:, :target_rank]
            per = max(1.5, min(8.0, (deadline - 15.0 - time.time()) / max(1, max_attempts - attempts)))
            res = try_optimize_rank(target_rank, time_budget=per,
                                    init_U=U_init, init_V=V_init, init_W=W_init,
                                    max_restarts=4, max_nfev=3000)
            if res is not None:
                best_U, best_V, best_W = res
                best_rank = target_rank
                found = True
                break
        if found:
            continue  # try even lower rank with remaining time
        # If not found via warm-start, try a couple of pure random restarts
        if time.time() < deadline - 15.0:
            per = max(2.0, min(10.0, (deadline - 15.0 - time.time()) / 2.0))
            res = try_optimize_rank(target_rank, time_budget=per,
                                    max_restarts=6, max_nfev=3000)
            if res is not None:
                best_U, best_V, best_W = res
                best_rank = target_rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
