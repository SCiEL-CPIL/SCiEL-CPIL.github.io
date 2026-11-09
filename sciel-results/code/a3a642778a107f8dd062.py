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

    # --- Strategy 1: Gaussian Process with multiple kernels optimized via L-BFGS-B ---
    if elapsed() < deadline:
        def matern52_kernel(D, ls):
            sqrt3d = np.sqrt(3.0) * D / ls
            return (1.0 + sqrt3d + sqrt3d**2 / 3.0) * np.exp(-sqrt3d)

        def rbf_gp_kernel(D, ls):
            return np.exp(-0.5 * (D / ls)**2)

        def rq_kernel(D, ls, alpha):
            return (1.0 + (D**2) / (2.0 * alpha * ls**2)) ** (-alpha)

        # Precompute distance matrix once
        D = np.abs(xs[:, None] - xs[None, :])
        D2 = D ** 2

        # Kernel types: 0=Matern52, 1=RBF, 2=RationalQuadratic
        kernel_types = [
            ("m52", lambda D, ls, a: matern52_kernel(D, ls)),
            ("rbf", lambda D, ls, a: rbf_gp_kernel(D, ls)),
            ("rq", lambda D, ls, a: rq_kernel(D, ls, a)),
        ]

        best_nll_overall = np.inf
        best_kernel_idx = 0
        best_alpha_overall = None
        best_params_overall = None

        for k_idx, (k_name, k_func) in enumerate(kernel_types):
            if elapsed() > deadline:
                break

            def gp_neg_log_likelihood(params, k_idx=k_idx, k_func=k_func):
                if k_idx == 2:  # RQ has 3 params
                    log_ls, log_sigma2, log_alpha = params
                    ls = np.exp(log_ls)
                    sigma2 = np.exp(log_sigma2)
                    alpha = np.exp(log_alpha)
                    K = sigma2 * k_func(D, ls, alpha) + 1e-10 * np.eye(n)
                else:
                    log_ls, log_sigma2 = params
                    ls = np.exp(log_ls)
                    sigma2 = np.exp(log_sigma2)
                    K = sigma2 * k_func(D, ls, 1.0) + 1e-10 * np.eye(n)

                try:
                    L = np.linalg.cholesky(K)
                    v = np.linalg.solve(L, ys)
                    alpha_vec = np.linalg.solve(L.T, v)
                    log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
                    y_alpha = float(ys @ alpha_vec)
                    nll = 0.5 * y_alpha + 0.5 * log_det_K + 0.5 * n * np.log(2 * np.pi)
                    if not np.isfinite(nll):
                        return 1e10
                    return nll
                except Exception:
                    return 1e10

            # Multi-start for this kernel
            ls_starts = np.logspace(np.log10(max(x_range / 50, 1e-6)), np.log10(x_range * 5), 6)
            sigma_starts = np.logspace(np.log10(max(y_var * 0.01, 1e-10)), np.log10(y_var * 50), 4)

            for ls0 in ls_starts:
                if elapsed() > deadline:
                    break
                for sig0 in sigma_starts:
                    if elapsed() > deadline:
                        break
                    try:
                        if k_idx == 2:
                            x0 = [np.log(ls0), np.log(sig0), np.log(2.0)]
                            bounds = [(np.log(max(x_range / 500, 1e-8)), np.log(x_range * 50)),
                                      (np.log(max(y_var * 0.001, 1e-12)), np.log(y_var * 500)),
                                      (np.log(0.1), np.log(20.0))]
                        else:
                            x0 = [np.log(ls0), np.log(sig0)]
                            bounds = [(np.log(max(x_range / 500, 1e-8)), np.log(x_range * 50)),
                                      (np.log(max(y_var * 0.001, 1e-12)), np.log(y_var * 500))]

                        res = minimize(gp_neg_log_likelihood, x0, method='L-BFGS-B',
                                       bounds=bounds, options={'maxiter': 80, 'ftol': 1e-10})

                        if res.fun < best_nll_overall:
                            best_nll_overall = res.fun
                            best_kernel_idx = k_idx
                            best_params_overall = res.x
                            # Recompute alpha
                            if k_idx == 2:
                                ls_b = np.exp(res.x[0]); sig2_b = np.exp(res.x[1]); a_b = np.exp(res.x[2])
                                K_b = sig2_b * rq_kernel(D, ls_b, a_b) + 1e-10 * np.eye(n)
                            else:
                                ls_b = np.exp(res.x[0]); sig2_b = np.exp(res.x[1])
                                K_b = sig2_b * k_func(D, ls_b, 1.0) + 1e-10 * np.eye(n)
                            L_b = np.linalg.cholesky(K_b)
                            v_b = np.linalg.solve(L_b, ys)
                            best_alpha_overall = np.linalg.solve(L_b.T, v_b)
                    except Exception:
                        pass

        if best_alpha_overall is not None and best_params_overall is not None:
            k_name_best, k_func_best = kernel_types[best_kernel_idx]
            if best_kernel_idx == 2:
                ls_b = np.exp(best_params_overall[0]); sig2_b = np.exp(best_params_overall[1]); a_b = np.exp(best_params_overall[2])
                def pred_gp(Xp, xs=xs, alpha=best_alpha_overall, ls=ls_b, sigma2=sig2_b, a=a_b):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Dp = np.abs(Xp[:, None] - xs[None, :])
                    Ks = sigma2 * rq_kernel(Dp, ls, a)
                    return Ks @ alpha
            else:
                ls_b = np.exp(best_params_overall[0]); sig2_b = np.exp(best_params_overall[1])
                def pred_gp(Xp, xs=xs, alpha=best_alpha_overall, ls=ls_b, sigma2=sig2_b, k_func=k_func_best):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Dp = np.abs(Xp[:, None] - xs[None, :])
                    Ks = sigma2 * k_func(Dp, ls, 1.0)
                    return Ks @ alpha

            add_candidate(f"gp_{k_name_best}", pred_gp)

    # --- Strategy 2: Kernel Ridge Regression with multiple widths ---
    if elapsed() < deadline:
        def rbf_kernel(X1, X2, width):
            D2 = (X1[:, None] - X2[None, :])**2
            return np.exp(-D2 / (2 * width**2))

        widths = np.logspace(np.log10(max(x_range / 100, 1e-4)), np.log10(x_range * 2), 12)
        for width in widths:
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

                add_candidate(f"krb_w{width:.4f}", pred_rbf)
            except Exception:
                pass

    # --- Strategy 3: Spline Interpolation with various smoothing ---
    if elapsed() < deadline:
        try:
            from scipy.interpolate import UnivariateSpline
            # Scale s relative to data variance for better defaults
            s_scale = y_var * n
            s_list = [0.0, s_scale * 1e-6, s_scale * 1e-5, s_scale * 1e-4,
                      s_scale * 1e-3, s_scale * 1e-2, s_scale * 0.1, s_scale * 1.0]
            for s in s_list:
                if elapsed() > deadline:
                    break
                try:
                    spl = UnivariateSpline(xs, ys, s=s)
                    def pred_spline(Xp, spl=spl):
                        Xp = np.asarray(Xp, dtype=float).reshape(-1)
                        return spl(Xp)
                    add_candidate(f"spline_s{s:.4f}", pred_spline)
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

    # --- Strategy 5: Fourier Basis with adaptive frequencies ---
    omegas = [2 * np.pi / x_range, np.pi / x_range, 4 * np.pi / x_range,
              2 * np.pi / (x_range * 2), np.pi / (x_range * 2)]
    for nf in range(1, 10):
        if elapsed() > deadline:
            break
        for omega in omegas:
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

    # --- Strategy 6: Local Polynomial Regression (weighted) ---
    if elapsed() < deadline:
        from scipy.stats import gaussian_kde
        # Use a few bandwidths for local polynomial fits
        for bw_factor in [0.5, 1.0, 2.0, 4.0]:
            if elapsed() > deadline:
                break
            try:
                # Estimate local bandwidth from data density
                try:
                    kde = gaussian_kde(xs)
                    # Use median density to set local bandwidth
                    med_density = np.median(kde(xs))
                    local_bw = 1.0 / max(med_density, 1e-10) * bw_factor
                except Exception:
                    local_bw = x_range / 10.0 * bw_factor

                def pred_localpoly(Xp, xs=xs, ys=ys, local_bw=local_bw, x_min=x_min, x_max=x_max):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    out = np.zeros(len(Xp))
                    for i, xi in enumerate(Xp):
                        # Weight by distance
                        d = np.abs(xs - xi)
                        wts = np.exp(-0.5 * (d / local_bw)**2)
                        # Only use nearby points
                        mask = wts > 1e-6
                        if np.sum(mask) < 3:
                            out[i] = np.mean(ys)
                            continue
                        xs_local = xs[mask]
                        ys_local = ys[mask]
                        wts_local = wts[mask]
                        # Fit local polynomial (degree 3)
                        xs_c = (xs_local - xi) / local_bw
                        V = np.vander(xs_c, N=4, increasing=True)
                        Vw = V * wts_local[:, None]
                        try:
                            wcoef, *_ = np.linalg.lstsq(Vw, ys_local, rcond=None)
                            out[i] = wcoef[0]
                        except Exception:
                            out[i] = np.average(ys_local, weights=wts_local)
                    return out

                add_candidate(f"localpoly_bw{bw_factor}", pred_localpoly)
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
