# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    x = X[:, 0].copy()
    x_min = float(x.min())
    x_max = float(x.max())
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_min = x[0] - 0.5 if n > 0 else 0.0
        x_max = x[0] + 0.5 if n > 0 else 1.0

    # Sort by x for numerical stability
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    y_var = float(np.var(y))
    if y_var < 1e-12:
        y_var = 1.0
    y_mean = float(np.mean(y))

    def elapsed():
        return time.time() - start_time

    candidates = []

    def add_candidate(name, pred_func):
        if elapsed() > deadline:
            return
        try:
            yp = np.asarray(pred_func(X), dtype=float).flatten()
            if yp.shape[0] != n or not np.all(np.isfinite(yp)):
                return
            mse = float(np.mean((yp - y) ** 2))
            if np.isfinite(mse):
                candidates.append((name, pred_func, mse))
        except Exception:
            pass

    # --- Strategy 1: GP with Matern 5/2 kernel, L-BFGS-B optimization ---
    if elapsed() < deadline:
        def matern52_kernel(D, ls):
            sqrt3d = np.sqrt(3.0) * D / ls
            return (1.0 + sqrt3d + sqrt3d**2 / 3.0) * np.exp(-sqrt3d)

        def gp_neg_log_likelihood(log_params):
            log_ls = log_params[0]
            log_sigma = log_params[1]
            ls = np.exp(log_ls)
            sigma2 = np.exp(log_sigma)
            try:
                D = np.sqrt(np.sum((xs[:, None] - xs[None, :])**2, axis=2))
                K = sigma2 * matern52_kernel(D, ls) + 1e-10 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))
                nll = 0.5 * float(ys @ alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
                if not np.isfinite(nll):
                    return 1e10
                return nll
            except Exception:
                return 1e10

        # Grid of starting points for ls and sigma2
        ls_starts = [x_range / 3.0, x_range / 5.0, x_range / 8.0, x_range / 15.0,
                     x_range / 25.0, 0.1, 0.2, 0.5, 1.0, 2.0]
        sigma_starts = [y_var, y_var * 0.5, y_var * 2.0]

        best_nll = np.inf
        best_params = None

        for ls0 in ls_starts:
            if elapsed() > deadline:
                break
            for sigma0 in sigma_starts:
                if elapsed() > deadline:
                    break
                try:
                    x0 = np.array([np.log(max(ls0, 1e-4)), np.log(max(sigma0, 1e-10))])
                    res = minimize(gp_neg_log_likelihood, x0, method='L-BFGS-B',
                                   bounds=[(-5, 5), (-5, 12)],
                                   options={'maxiter': 50, 'ftol': 1e-8})
                    if res.fun < best_nll:
                        best_nll = res.fun
                        best_params = res.x
                except Exception:
                    pass

        if best_params is not None:
            ls_opt = float(np.exp(best_params[0]))
            sigma2_opt = float(np.exp(best_params[1]))

            try:
                D = np.sqrt(np.sum((xs[:, None] - xs[None, :])**2, axis=2))
                K = sigma2_opt * matern52_kernel(D, ls_opt) + 1e-10 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))

                def pred_gp(Xp, xs=xs, L=L, alpha=alpha, ls=ls_opt, sigma2=sigma2_opt, matern52_kernel=matern52_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    D = np.sqrt(np.sum((Xp[:, None] - xs[None, :])**2, axis=1))
                    Ks = sigma2 * matern52_kernel(D, ls)
                    return Ks @ alpha

                add_candidate("gp_opt", pred_gp)
            except Exception:
                pass

        # Also try a few fixed ls values without optimization (faster)
        for ls in [x_range / 3.0, x_range / 5.0, x_range / 8.0, x_range / 15.0, 0.1, 0.2, 0.5, 1.0]:
            if ls <= 0 or elapsed() > deadline:
                continue
            try:
                D = np.sqrt(np.sum((xs[:, None] - xs[None, :])**2, axis=2))
                K = y_var * matern52_kernel(D, ls) + 1e-10 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))

                def pred_gp_fixed(Xp, xs=xs, L=L, alpha=alpha, ls=ls, sigma2=y_var, matern52_kernel=matern52_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    D = np.sqrt(np.sum((Xp[:, None] - xs[None, :])**2, axis=1))
                    Ks = sigma2 * matern52_kernel(D, ls)
                    return Ks @ alpha

                add_candidate(f"gp_ls{ls:.4f}", pred_gp_fixed)
            except Exception:
                pass

    # --- Strategy 2: RBF kernel regression with optimized width ---
    if elapsed() < deadline:
        def rbf_kernel(X1, X2, width):
            D2 = np.sum((X1[:, None] - X2[None, :])**2, axis=2)
            return np.exp(-D2 / (2 * width**2))

        for width in [x_range / 4.0, x_range / 6.0, x_range / 10.0, x_range / 20.0,
                      x_range / 40.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
            if width <= 0 or elapsed() > deadline:
                continue
            try:
                K = rbf_kernel(xs, xs, width)
                K += 1e-8 * np.eye(n)
                w, *_ = np.linalg.lstsq(K, ys, rcond=None)

                def pred_rbf(Xp, xs=xs, w=w, width=width, rbf_kernel=rbf_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Ks = rbf_kernel(Xp, xs, width)
                    return Ks @ w

                add_candidate(f"rbf_w{width:.4f}", pred_rbf)
            except Exception:
                pass

    # --- Strategy 3: Spline interpolation ---
    if elapsed() < deadline:
        for s in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            if elapsed() > deadline:
                break
            try:
                spl = UnivariateSpline(xs, ys, s=s)
                def pred_spline(Xp, spl=spl):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    return spl(Xp)
                add_candidate(f"spline_s{s}", pred_spline)
            except Exception:
                pass

    # --- Strategy 4: Polynomial regression ---
    x_scaled = (x - x_min) / x_range
    for deg in range(1, 15):
        if elapsed() > deadline:
            break
        try:
            V = np.vander(x_scaled, N=deg + 1, increasing=True)
            w, *_ = np.linalg.lstsq(V, y, rcond=None)

            def pred_poly(Xp, deg=deg, w=w, x_min=x_min, x_range=x_range):
                Xp = np.asarray(Xp, dtype=float).reshape(-1)
                Xs = (Xp - x_min) / x_range
                Vp = np.vander(Xs, N=deg + 1, increasing=True)
                return Vp @ w

            add_candidate(f"poly_{deg}", pred_poly)
        except Exception:
            pass

    # --- Strategy 5: Fourier basis ---
    omegas = [np.pi / x_range, 2 * np.pi / x_range, np.pi, 2 * np.pi, 1.0, 0.5, 0.25, 4.0, 0.1, 5.0]
    for nf in range(1, 10):
        if elapsed() > deadline:
            break
        for omega in omegas[:5]:  # Limit to first 5 omegas for speed
            if elapsed() > deadline:
                break
            try:
                cols = [np.ones(n)]
                for k in range(1, nf + 1):
                    cols.append(np.sin(k * omega * x))
                    cols.append(np.cos(k * omega * x))
                V = np.column_stack(cols)
                w, *_ = np.linalg.lstsq(V, y, rcond=None)

                def pred_fourier(Xp, nf=nf, omega=omega, w=w):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    cols = [np.ones(len(Xp))]
                    for k in range(1, nf + 1):
                        cols.append(np.sin(k * omega * Xp))
                        cols.append(np.cos(k * omega * Xp))
                    Vp = np.column_stack(cols)
                    return Vp @ w

                add_candidate(f"four_nf{nf}_om{omega:.4f}", pred_fourier)
            except Exception:
                pass

    # --- Strategy 6: Rich basis with greedy selection ---
    if elapsed() < deadline:
        basis_funcs = {}
        basis_funcs['1'] = lambda x: np.ones_like(x)
        basis_funcs['x'] = lambda x: x
        basis_funcs['x2'] = lambda x: x ** 2
        basis_funcs['x3'] = lambda x: x ** 3
        basis_funcs['x4'] = lambda x: x ** 4
        basis_funcs['abs_x'] = lambda x: np.abs(x)
        basis_funcs['sqrt_abs'] = lambda x: np.sqrt(np.abs(x))
        basis_funcs['log1p_abs'] = lambda x: np.log1p(np.abs(x))
        basis_funcs['tanh'] = lambda x: np.tanh(x)
        basis_funcs['tanh_2x'] = lambda x: np.tanh(2 * x)
        basis_funcs['exp_05x'] = lambda x: np.exp(0.5 * x)
        basis_funcs['exp_neg05x'] = lambda x: np.exp(-0.5 * x)
        basis_funcs['exp_01x'] = lambda x: np.exp(0.1 * x)
        basis_funcs['exp_neg01x'] = lambda x: np.exp(-0.1 * x)
        basis_funcs['sin_x'] = lambda x: np.sin(x)
        basis_funcs['cos_x'] = lambda x: np.cos(x)
        basis_funcs['sin_2x'] = lambda x: np.sin(2 * x)
        basis_funcs['cos_2x'] = lambda x: np.cos(2 * x)
        basis_funcs['sin_pi_x'] = lambda x: np.sin(np.pi * x)
        basis_funcs['cos_pi_x'] = lambda x: np.cos(np.pi * x)
        basis_funcs['exp_negx2'] = lambda x: np.exp(-x ** 2)
        basis_funcs['exp_neg05x2'] = lambda x: np.exp(-0.5 * x ** 2)

        basis_names = list(basis_funcs.keys())
        B = np.zeros((n, len(basis_names)))
        for j, name in enumerate(basis_names):
            try:
                B[:, j] = basis_funcs[name](x)
                B[:, j] = np.where(np.isfinite(B[:, j]), B[:, j], 0.0)
            except Exception:
                pass

        if elapsed() < deadline:
            selected = []
            remaining = list(range(len(basis_names)))
            for step in range(min(8, len(basis_names))):
                if elapsed() > deadline:
                    break
                best_j = None
                best_rss = np.inf
                for j in remaining:
                    try:
                        if selected:
                            V = np.column_stack([B[:, i] for i in selected] + [B[:, j]])
                        else:
                            V = B[:, j:j+1]
                        w, *_ = np.linalg.lstsq(V, y, rcond=None)
                        rss = float(np.sum((V @ w - y) ** 2))
                        if np.isfinite(rss) and rss < best_rss:
                            best_rss = rss
                            best_j = j
                    except Exception:
                        pass
                if best_j is None:
                    break
                selected.append(best_j)
                remaining.remove(best_j)

                sel_names = [basis_names[i] for i in selected]
                try:
                    V = np.column_stack([B[:, i] for i in selected])
                    w, *_ = np.linalg.lstsq(V, y, rcond=None)

                    def pred_sel(Xp, sel_names=sel_names, w=w):
                        Xp = np.asarray(Xp, dtype=float).reshape(-1)
                        cols = []
                        for nm in sel_names:
                            c = basis_funcs[nm](Xp)
                            c = np.where(np.isfinite(c), c, 0.0)
                            cols.append(c)
                        Vp = np.column_stack(cols)
                        return Vp @ w

                    add_candidate(f"sel_{'_'.join(sel_names[:3])}", pred_sel)
                except Exception:
                    pass

    # Select best candidate
    if not candidates:
        # Fallback: linear regression
        A = np.column_stack([np.ones(n), x])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)

        def pred_lin(Xp, w=w):
            Xp = np.asarray(Xp, dtype=float).reshape(-1)
            return np.column_stack([np.ones(len(Xp)), Xp]) @ w

        return pred_lin

    candidates.sort(key=lambda t: t[2])
    best_name, best_pred, best_mse = candidates[0]

    def final_predict(Xp):
        out = best_pred(Xp)
        out = np.asarray(out, dtype=float).flatten()
        out = np.where(np.isfinite(out), out, 0.0)
        return out

    return final_predict
# EVOLVE-BLOCK-END
