# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit

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

    # --- Strategy 1: Gaussian Process with Matern 5/2 kernel optimized via L-BFGS-B ---
    if elapsed() < deadline:
        def matern52_kernel(D, ls):
            sqrt3d = np.sqrt(3.0) * D / ls
            return (1.0 + sqrt3d + sqrt3d**2 / 3.0) * np.exp(-sqrt3d)

        # Precompute distance matrix once
        D = np.abs(xs[:, None] - xs[None, :])
        
        def gp_neg_log_likelihood(params):
            log_ls, log_sigma2 = params
            ls = np.exp(log_ls)
            sigma2 = np.exp(log_sigma2)
            
            try:
                K = sigma2 * matern52_kernel(D, ls) + 1e-10 * np.eye(n)
                L = np.linalg.cholesky(K)
                
                # Compute alpha
                v = np.linalg.solve(L, ys)
                alpha = np.linalg.solve(L.T, v)
                
                # Negative log marginal likelihood
                # -0.5 * y^T K^{-1} y - 0.5 * log det K - n/2 * log(2*pi)
                log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
                y_alpha = float(ys @ alpha)
                nll = 0.5 * y_alpha + 0.5 * log_det_K + 0.5 * n * np.log(2 * np.pi)
                
                if not np.isfinite(nll):
                    return 1e10, None
                    
                # Store alpha for prediction if this is the best
                return nll, alpha
            except Exception:
                return 1e10, None

        # Multi-start optimization over log-transformed parameters
        # ls in [x_range/100, x_range*10], sigma2 in [y_var*0.01, y_var*100]
        ls_starts = np.logspace(np.log10(x_range / 100), np.log10(x_range * 10), 8)
        sigma_starts = np.logspace(np.log10(max(y_var * 0.01, 1e-10)), np.log10(y_var * 100), 5)
        
        best_nll = np.inf
        best_params = None
        best_alpha = None
        
        for ls0 in ls_starts:
            if elapsed() > deadline:
                break
            for sig0 in sigma_starts:
                if elapsed() > deadline:
                    break
                try:
                    x0 = [np.log(ls0), np.log(sig0)]
                    bounds = [(np.log(x_range / 1000), np.log(x_range * 100)),
                              (np.log(max(y_var * 0.001, 1e-12)), np.log(y_var * 1000))]
                    
                    res = minimize(gp_neg_log_likelihood, x0, method='L-BFGS-B',
                                   bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
                    
                    if res.fun < best_nll:
                        best_nll = res.fun
                        best_params = res.x
                        # Recompute alpha with best params
                        ls_best = np.exp(best_params[0])
                        sigma2_best = np.exp(best_params[1])
                        K_best = sigma2_best * matern52_kernel(D, ls_best) + 1e-10 * np.eye(n)
                        L_best = np.linalg.cholesky(K_best)
                        v_best = np.linalg.solve(L_best, ys)
                        best_alpha = np.linalg.solve(L_best.T, v_best)
                except Exception:
                    pass

        if best_params is not None and best_alpha is not None:
            ls_best = np.exp(best_params[0])
            sigma2_best = np.exp(best_params[1])
            
            def pred_gp(Xp, xs=xs, alpha=best_alpha, ls=ls_best, sigma2=sigma2_best):
                Xp = np.asarray(Xp, dtype=float).reshape(-1)
                Dp = np.abs(Xp[:, None] - xs[None, :])
                Ks = sigma2 * matern52_kernel(Dp, ls)
                return Ks @ alpha

            add_candidate("gp_optimized", pred_gp)

    # --- Strategy 2: RBF Kernel Regression (Kernel Ridge) ---
    if elapsed() < deadline:
        def rbf_kernel(X1, X2, width):
            D2 = (X1[:, None] - X2[None, :])**2
            return np.exp(-D2 / (2 * width**2))

        for width in [x_range / 2.0, x_range / 4.0, x_range / 8.0, x_range / 16.0,
                      x_range / 32.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]:
            if width <= 0 or elapsed() > deadline:
                continue
            try:
                K = rbf_kernel(xs, xs, width)
                K += 1e-6 * np.eye(n)
                w, *_ = np.linalg.lstsq(K, ys, rcond=None)

                def pred_rbf(Xp, xs=xs, w=w, width=width, rbf_kernel=rbf_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Ks = rbf_kernel(Xp, xs, width)
                    return Ks @ w

                add_candidate(f"rbf_w{width:.3f}", pred_rbf)
            except Exception:
                pass

    # --- Strategy 3: Spline Interpolation ---
    if elapsed() < deadline:
        try:
            from scipy.interpolate import UnivariateSpline
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
        except ImportError:
            pass

    # --- Strategy 4: Polynomial Regression ---
    x_scaled = (x - x_min) / x_range
    for deg in range(1, 12):
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

    # --- Strategy 5: Fourier Basis ---
    omegas = [np.pi / x_range, 2 * np.pi / x_range, np.pi, 2 * np.pi, 1.0, 0.5, 2.0, 4.0]
    for nf in range(1, 8):
        if elapsed() > deadline:
            break
        for omega in omegas[:4]:
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

                add_candidate(f"four_nf{nf}_om{omega:.3f}", pred_fourier)
            except Exception:
                pass

    # Select best candidate
    if not candidates:
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
