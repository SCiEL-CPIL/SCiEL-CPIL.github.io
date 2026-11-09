"""Symbolic regression: Gaussian Process (Cholesky) + adaptive forward-selection basis.

Structural idea (from the GP inspiration): treat the noiseless 1D problem as a
Gaussian process with an RBF kernel and a tiny noise floor, solve via Cholesky,
and optimize length-scale / signal-variance with L-BFGS-B. This gives exact
interpolation of smooth functions with smooth generalization between points,
avoiding the basis-truncation error of fixed-basis fits.

Refinement machinery (from the current program): a rich candidate basis
(polynomials, trig, exp, RBF, rational, sqrt/abs/tanh/log), a ridge solve,
forward selection with 5-fold CV, and a centered-polynomial fallback. The GP and
the basis model are scored with the SAME CV harness and the better one is kept.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')


def _safe_predict(predict_fn, X):
    """Ensure predictions are finite."""
    X = np.asarray(X, dtype=float)
    try:
        pred = predict_fn(X)
    except Exception:
        return np.zeros(len(np.asarray(X).reshape(-1)))
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return pred


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin well under the 330s kill

    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)

    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_vals = X[:, 0].copy()
    y_vals = y.copy()

    # Constant check
    if np.std(y_vals) < 1e-12:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    y_var = np.var(y_vals)
    if y_var < 1e-15:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    x_center = (x_min + x_max) / 2.0

    # 5-fold CV setup
    rng = np.random.RandomState(42)
    idx_all = np.arange(n)
    rng.shuffle(idx_all)
    folds = np.array_split(idx_all, 5)

    def cv_mse(basis_matrix, y_target):
        """Compute CV normalized MSE for a given basis matrix."""
        if time.time() > deadline - 3:
            return np.inf
        try:
            se = 0.0
            for tr in range(5):
                train_idx = np.concatenate([folds[i] for i in range(5) if i != tr])
                val_idx = folds[tr]
                A_tr = basis_matrix[train_idx]
                b_tr = y_target[train_idx]
                A_va = basis_matrix[val_idx]
                b_va = y_target[val_idx]
                n_cols = A_tr.shape[1]
                try:
                    w = np.linalg.solve(A_tr.T @ A_tr + 1e-6 * np.eye(n_cols), A_tr.T @ b_tr)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(A_tr, b_tr, rcond=None)[0]
                pred = A_va @ w
                if not np.all(np.isfinite(pred)):
                    return np.inf
                se += np.sum((pred - b_va) ** 2)
            return se / (n * y_var)
        except Exception:
            return np.inf

    # ------------------------------------------------------------------
    # Gaussian Process model (RBF kernel, Cholesky, tiny noise floor)
    # ------------------------------------------------------------------
    def _gp_predict(Xq, x_tr, y_tr, alpha, ls, sv):
        Xq = np.asarray(Xq, dtype=float).reshape(-1, 1)
        sq = np.sum(Xq ** 2, axis=1)[:, None] + np.sum(x_tr ** 2, axis=1)[None, :] - 2.0 * (Xq @ x_tr.T)
        sq = np.maximum(sq, 0.0)
        K = sv * np.exp(-sq / (2.0 * ls ** 2))
        return K @ alpha

    def _optimize_gp():
        """Optimize RBF GP hyperparameters; return (ls, sv, alpha, cv_score) or None."""
        x_tr = x_vals
        y_tr = y_vals

        def _neg_log_marginal_likelihood(params):
            if time.time() > deadline - 6:
                return 1e10
            log_ls, log_sv = params
            ls = np.exp(log_ls)
            sv = np.exp(log_sv)
            nv = max(1e-12, y_var * 1e-12)
            if ls < 1e-8 or sv < 1e-12:
                return 1e10
            try:
                sq = np.sum(x_tr ** 2, axis=1)[:, None] + np.sum(x_tr ** 2, axis=1)[None, :] - 2.0 * (x_tr @ x_tr.T)
                sq = np.maximum(sq, 0.0)
                K = sv * np.exp(-sq / (2.0 * ls ** 2))
                K[np.diag_indices_from(K)] += nv
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L, y_tr)
                alpha = np.linalg.solve(L.T, alpha)
                log_lik = -0.5 * y_tr @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                if not np.isfinite(log_lik):
                    return 1e10
                return -log_lik
            except Exception:
                return 1e10

        best_neg_ll = np.inf
        best_params = None
        ls_cands = [x_range * f for f in [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]]
        sv_cands = [y_var * f for f in [0.5, 1.0, 2.0, 4.0]]
        max_inits = 12
        n_inits = 0
        for ls_c in ls_cands:
            for sv_c in sv_cands:
                if n_inits >= max_inits or time.time() > deadline - 25:
                    break
                x_init = np.log([max(ls_c, 1e-6), max(sv_c, 1e-10)])
                try:
                    res = minimize(
                        _neg_log_marginal_likelihood,
                        x_init,
                        method='L-BFGS-B',
                        bounds=[(np.log(1e-5), np.log(x_range * 5)),
                                (np.log(1e-10), np.log(y_var * 50))],
                        options={'maxiter': 25, 'ftol': 1e-5, 'gtol': 1e-3},
                    )
                    if np.isfinite(res.fun) and res.fun < best_neg_ll:
                        best_neg_ll = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass
                n_inits += 1
            if time.time() > deadline - 25:
                break

        if best_params is None:
            return None

        ls = float(np.clip(np.exp(best_params[0]), 1e-4, x_range * 5))
        sv = float(np.clip(np.exp(best_params[1]), 1e-8, y_var * 50))
        nv = max(1e-12, y_var * 1e-12)
        try:
            sq = np.sum(x_tr ** 2, axis=1)[:, None] + np.sum(x_tr ** 2, axis=1)[None, :] - 2.0 * (x_tr @ x_tr.T)
            sq = np.maximum(sq, 0.0)
            K = sv * np.exp(-sq / (2.0 * ls ** 2))
            K[np.diag_indices_from(K)] += nv
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L, y_tr)
            alpha = np.linalg.solve(L.T, alpha)
            if not np.all(np.isfinite(alpha)):
                return None
        except Exception:
            return None

        # CV score for the GP (train GP on 4 folds, predict the held-out fold)
        try:
            se = 0.0
            for tr in range(5):
                if time.time() > deadline - 4:
                    return None
                train_idx = np.concatenate([folds[i] for i in range(5) if i != tr])
                val_idx = folds[tr]
                xt = x_tr[train_idx]
                yt = y_tr[train_idx]
                sq = np.sum(xt ** 2, axis=1)[:, None] + np.sum(xt ** 2, axis=1)[None, :] - 2.0 * (xt @ xt.T)
                sq = np.maximum(sq, 0.0)
                Kt = sv * np.exp(-sq / (2.0 * ls ** 2))
                Kt[np.diag_indices_from(Kt)] += nv
                Lt = np.linalg.cholesky(Kt)
                at = np.linalg.solve(Lt, yt)
                at = np.linalg.solve(Lt.T, at)
                pred = _gp_predict(x_tr[val_idx], xt, yt, at, ls, sv)
                if not np.all(np.isfinite(pred)):
                    return None
                se += np.sum((pred - y_tr[val_idx]) ** 2)
            return (ls, sv, alpha, se / (n * y_var))
        except Exception:
            return None

    gp_result = None
    if time.time() < deadline - 30:
        gp_result = _optimize_gp()

    # ------------------------------------------------------------------
    # Adaptive basis model (forward selection) + polynomial fallback
    # ------------------------------------------------------------------
    def make_basis_funcs():
        funcs = []
        funcs.append(('const', lambda x: np.ones_like(x, dtype=float)))
        funcs.append(('x', lambda x: x.copy()))
        for d in range(2, 10):
            funcs.append((f'c^{d}', lambda x, d=d: (x - x_center) ** d))
        for f in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
            funcs.append((f'cos{f}', lambda x, f=f: np.cos(f * x)))
            funcs.append((f'sin{f}', lambda x, f=f: np.sin(f * x)))
        for b in [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]:
            funcs.append((f'exp{b}', lambda x, b=b: np.exp(b * x)))
        n_centers = 12
        centers = np.linspace(x_min, x_max, n_centers + 2)[1:-1]
        widths = [0.1 * x_range, 0.2 * x_range, 0.3 * x_range, 0.5 * x_range]
        for c in centers:
            for w in widths:
                funcs.append((f'rbf_{c:.3f}_{w:.3f}', lambda x, c=c, w=w: np.exp(-((x - c) ** 2) / (2 * w ** 2))))
        funcs.append(('1/(x+1)', lambda x: 1.0 / (x + 1.0 + 1e-10)))
        funcs.append(('1/(x-1)', lambda x: 1.0 / (x - 1.0 + 1e-10)))
        funcs.append(('1/(x^2+1)', lambda x: 1.0 / (x ** 2 + 1.0)))
        funcs.append(('1/(x^2+0.1)', lambda x: 1.0 / (x ** 2 + 0.1)))
        funcs.append(('abs', lambda x: np.abs(x)))
        funcs.append(('x| x|', lambda x: x * np.abs(x)))
        funcs.append(('sqrt(x^2+1)', lambda x: np.sqrt(x ** 2 + 1.0)))
        if x_min > -0.1:
            funcs.append(('sqrt(x+0.1)', lambda x: np.sqrt(np.maximum(x + 0.1, 1e-10))))
        for f in [0.25, 0.5, 1.0, 2.0, 4.0]:
            funcs.append((f'tanh{f}', lambda x, f=f: np.tanh(f * x)))
        funcs.append(('log(x^2+1)', lambda x: np.log(x ** 2 + 1.0)))
        if x_min > -0.1:
            funcs.append(('log(x+0.1)', lambda x: np.log(np.maximum(x + 0.1, 1e-10))))
        return funcs

    candidate_pool = make_basis_funcs()
    n_cand = len(candidate_pool)
    B_all = np.zeros((n, n_cand))
    valid_cand = []
    for j, (name, func) in enumerate(candidate_pool):
        try:
            col = np.asarray(func(x_vals), dtype=float)
            if not np.all(np.isfinite(col)):
                col = np.where(np.isfinite(col), col, 0.0)
            if np.var(col) > 1e-10:
                B_all[:, j] = col
                valid_cand.append(j)
        except Exception:
            pass

    valid_cand = np.array(valid_cand)
    B_valid = B_all[:, valid_cand]
    n_valid = len(valid_cand)

    selected = [0]
    best_cv = cv_mse(B_valid[:, selected], y_vals)
    max_iter = min(25, n_valid)
    for it in range(max_iter):
        if time.time() > deadline - 12:
            break
        best_add_idx = -1
        best_add_cv = best_cv
        for j in range(n_valid):
            if j in selected:
                continue
            trial_selected = selected + [j]
            cv = cv_mse(B_valid[:, trial_selected], y_vals)
            if cv < best_add_cv:
                best_add_cv = cv
                best_add_idx = j
        if best_add_idx >= 0:
            selected.append(best_add_idx)
            best_cv = best_add_cv
            if best_cv < 1e-6:
                break
        else:
            break

    A_final = B_valid[:, selected]
    n_cols = A_final.shape[1]
    try:
        w_final = np.linalg.solve(A_final.T @ A_final + 1e-6 * np.eye(n_cols), A_final.T @ y_vals)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(A_final, y_vals, rcond=None)[0]
    if not np.all(np.isfinite(w_final)):
        w_final = np.zeros(n_cols)
        w_final[0] = np.mean(y_vals)

    # Polynomial fallback
    poly_best = None
    poly_best_cv = np.inf
    for deg in range(1, min(12, n)):
        if time.time() > deadline - 6:
            break
        try:
            A_poly = np.zeros((n, deg + 1))
            for d in range(deg + 1):
                A_poly[:, d] = (x_vals - x_center) ** d
            cv = cv_mse(A_poly, y_vals)
            if cv < poly_best_cv:
                poly_best_cv = cv
                try:
                    w_poly = np.linalg.solve(A_poly.T @ A_poly + 1e-6 * np.eye(deg + 1), A_poly.T @ y_vals)
                except np.linalg.LinAlgError:
                    w_poly = np.linalg.lstsq(A_poly, y_vals, rcond=None)[0]
                if np.all(np.isfinite(w_poly)):
                    poly_best = (deg, w_poly)
        except Exception:
            continue

    # Choose the best model: GP vs basis vs polynomial (all on the same CV scale)
    gp_cv = gp_result[3] if gp_result is not None else np.inf
    use_gp = (gp_result is not None and gp_cv < best_cv * 0.98 and gp_cv < poly_best_cv * 0.98)
    use_poly = (not use_gp) and (poly_best is not None and poly_best_cv < best_cv * 0.95)

    gp_ls = gp_result[0] if gp_result is not None else None
    gp_sv = gp_result[1] if gp_result is not None else None
    gp_alpha = gp_result[2] if gp_result is not None else None

    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
        n_pred = len(X_pred)
        try:
            if use_gp:
                pred = _gp_predict(X_pred, x_vals, y_vals, gp_alpha, gp_ls, gp_sv)
            elif use_poly and poly_best is not None:
                deg, w_poly = poly_best
                A_pred = np.zeros((n_pred, deg + 1))
                for d in range(deg + 1):
                    A_pred[:, d] = (X_pred - x_center) ** d
                pred = A_pred @ w_poly
            else:
                A_pred = np.zeros((n_pred, len(selected)))
                for k, idx in enumerate(selected):
                    name, func = candidate_pool[idx]
                    try:
                        col = np.asarray(func(X_pred), dtype=float)
                        if not np.all(np.isfinite(col)):
                            col = np.where(np.isfinite(col), col, 0.0)
                        A_pred[:, k] = col
                    except Exception:
                        A_pred[:, k] = 0.0
                pred = A_pred @ w_final
            pred = np.asarray(pred, dtype=float)
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            return np.full(n_pred, np.mean(y_vals))

    return predict
# EVOLVE-BLOCK-END
