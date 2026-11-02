import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a0+a1)(b0+c1) -> contributes to c00, c11
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M2: (a1)(b0-b1) -> contributes to c01, c11
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        # M3: (a0)(b1-c1) -> contributes to c10, c11
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M4: (a1)(c0-b0) -> contributes to c00, c01
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        # M5: (a0+a1)(c1) -> contributes to c00, c10
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        # M6: (a1-a0)(b0+c1) -> contributes to c11
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M7: (a1)(b0+c1) -> contributes to c00
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W

def _build_true_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0
    return T

def _max_residual(U, V, W, T_true):
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.max(np.abs(M - T_true))

def _try_eliminate_column(U0, V0, W0, T_true, target_col, time_budget, seed=0):
    """Use smooth optimization to drive column target_col to zero."""
    start = time.time()
    nm, mp, pn = U0.shape[0], V0.shape[0], W0.shape[0]
    rank = U0.shape[1]
    
    rng = np.random.RandomState(seed)

    def params_to_mats(x):
        nU = nm * rank
        nV = mp * rank
        nW = pn * rank
        U = x[:nU].reshape(nm, rank)
        V = x[nU:nU+nV].reshape(mp, rank)
        W = x[nU+nV:nU+nV+nW].reshape(pn, rank)
        t = x[nU+nV+nW]
        return U, V, W, t

    def objective(x):
        U, V, W, t = params_to_mats(x)
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        resid = M - T_true
        r2 = resid ** 2
        max_r2 = np.max(r2)
        if max_r2 < 1e-30:
            soft_max = 0.0
        else:
            shifted = r2 - max_r2
            lse = max_r2 + np.log(np.sum(np.exp(shifted)))
            soft_max = np.sqrt(lse)
        col_norm_sq = (U[:, target_col] ** 2).sum() + (V[:, target_col] ** 2).sum() + (W[:, target_col] ** 2).sum()
        lam = 10.0
        penalty = 0.0
        if t < soft_max:
            penalty = 200.0 * (soft_max - t) ** 2
        return t + lam * col_norm_sq + penalty

    def gradient(x):
        U, V, W, t = params_to_mats(x)
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        resid = M - T_true
        r2 = resid ** 2
        max_r2 = np.max(r2)
        if max_r2 < 1e-30:
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            soft_max = 0.0
        else:
            shifted = r2 - max_r2
            weights = np.exp(shifted) / np.sum(np.exp(shifted))
            soft_max = np.sqrt(max_r2 + np.log(np.sum(np.exp(shifted))))
            if soft_max < 1e-15:
                d_resid = np.zeros_like(resid)
            else:
                d_resid = 2.0 * weights * resid / soft_max
            dU = np.einsum('ijk,jr,kr->ir', d_resid, V, W)
            dV = np.einsum('ijk,ir,kr->jr', d_resid, U, W)
            dW = np.einsum('ijk,ir,jr->kr', d_resid, U, V)

        lam = 10.0
        dU[:, target_col] += 2.0 * lam * U[:, target_col]
        dV[:, target_col] += 2.0 * lam * V[:, target_col]
        dW[:, target_col] += 2.0 * lam * W[:, target_col]

        dt = 1.0
        if t < soft_max:
            dt += 400.0 * (t - soft_max)

        nU = nm * rank
        nV = mp * rank
        nW = pn * rank
        gx = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten(), [dt]])
        return gx

    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten(), [1.0]])

    # Add small perturbation to target column to break symmetry
    if seed > 0:
        nU = nm * rank
        nV = mp * rank
        nW = pn * rank
        # Perturb target column in U, V, W
        u_start = 0
        v_start = nU
        w_start = nU + nV
        x0[u_start + target_col] += rng.randn() * 0.1
        x0[v_start + target_col] += rng.randn() * 0.1
        x0[w_start + target_col] += rng.randn() * 0.1

    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 1000})
    except Exception:
        return None, None, None, None

    if time.time() - start > time_budget:
        return None, None, None, None

    U_opt, V_opt, W_opt, t_opt = params_to_mats(res.x)
    col_norm = np.sqrt((U_opt[:, target_col] ** 2).sum() + (V_opt[:, target_col] ** 2).sum() + (W_opt[:, target_col] ** 2).sum())

    if col_norm < 1e-5:
        U31 = np.delete(U_opt, target_col, axis=1)
        V31 = np.delete(V_opt, target_col, axis=1)
        W31 = np.delete(W_opt, target_col, axis=1)
        M31 = np.einsum('ir,jr,kr->ijk', U31, V31, W31)
        max_res = np.max(np.abs(M31 - T_true))
        if max_res < 1e-7:
            # Final polish: refine with tighter tolerance
            U31 = U31.copy()
            V31 = V31.copy()
            W31 = W31.copy()
            try:
                def polish_obj(x):
                    U = x[:nm*31].reshape(nm, 31)
                    V = x[nm*31:nm*31+mp*31].reshape(mp, 31)
                    W = x[nm*31+mp*31:nm*31+mp*31+pn*31].reshape(pn, 31)
                    M = np.einsum('ir,jr,kr->ijk', U, V, W)
                    resid = M - T_true
                    return np.sum(resid**2)
                
                def polish_grad(x):
                    U = x[:nm*31].reshape(nm, 31)
                    V = x[nm*31:nm*31+mp*31].reshape(mp, 31)
                    W = x[nm*31+mp*31:nm*31+mp*31+pn*31].reshape(pn, 31)
                    M = np.einsum('ir,jr,kr->ijk', U, V, W)
                    resid = M - T_true
                    dU = np.einsum('ijk,jr,kr->ir', 2*resid, V, W)
                    dV = np.einsum('ijk,ir,kr->jr', 2*resid, U, W)
                    dW = np.einsum('ijk,ir,jr->kr', 2*resid, U, V)
                    return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                
                x_polish = np.concatenate([U31.flatten(), V31.flatten(), W31.flatten()])
                res_polish = minimize(polish_obj, x_polish, jac=polish_grad, method='L-BFGS-B',
                                     options={'maxiter': 200, 'ftol': 1e-18, 'gtol': 1e-14})
                U31 = res_polish.x[:nm*31].reshape(nm, 31)
                V31 = res_polish.x[nm*31:nm*31+mp*31].reshape(mp, 31)
                W31 = res_polish.x[nm*31+mp*31:nm*31+mp*31+pn*31].reshape(pn, 31)
                max_res = np.max(np.abs(np.einsum('ir,jr,kr->ijk', U31, V31, W31) - T_true))
            except Exception:
                pass
            
            if max_res < 1e-7:
                return U31, V31, W31, max_res

    return None, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS

    U32, V32, W32 = _build_rank32()
    T_true = _build_true_tensor(n, m, p)

    assert _max_residual(U32, V32, W32, T_true) < 1e-10, "Rank-32 solution is incorrect"

    best_U, best_V, best_W = U32.tolist(), V32.tolist(), W32.tolist()
    best_rank = 32

    rank = 32

    # Try each column with multiple seeds and shorter time budgets
    # Trivial block columns (28-31) are likely hardest; Strassen columns may be easier
    # Try Strassen columns first (0-27), then trivial (28-31)
    col_order = list(range(28)) + list(range(28, 32))
    
    for target_col in col_order:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 8.0:
            break

        # Try with seed 0 (no perturbation)
        time_per_attempt = min(20.0, (remaining - 8.0) / 2.0)
        U31, V31, W31, max_res = _try_eliminate_column(
            U32, V32, W32, T_true, target_col, time_per_attempt, seed=0
        )

        if U31 is not None and max_res < 1e-7:
            best_U = U31.tolist()
            best_V = V31.tolist()
            best_W = W31.tolist()
            best_rank = 31
            break

        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 8.0:
            break

        # Try with perturbation
        time_per_attempt = min(20.0, (remaining - 8.0))
        U31, V31, W31, max_res = _try_eliminate_column(
            U32, V32, W32, T_true, target_col, time_per_attempt, seed=42
        )

        if U31 is not None and max_res < 1e-7:
            best_U = U31.tolist()
            best_V = V31.tolist()
            best_W = W31.tolist()
            best_rank = 31
            break

    return {"U": best_U, "V": best_V, "W": best_W}
