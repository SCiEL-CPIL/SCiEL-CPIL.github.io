"""Symbolic regression via diverse basis search with cross-validation and adaptive refinement."""
import time
import numpy as np
from scipy.optimize import curve_fit, minimize
from itertools import product

# EVOLVE-BLOCK-START
import numpy as np

def fit(X_train, y_train):
    """
    Fit a regression model and return a predict callable.
    Uses cross-validation to select the best model from many candidates,
    with adaptive basis expansion and high-precision polish.
    """
    start_time = time.time()
    deadline = start_time + 285  # 15s safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0]

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    # Simple k-fold cross-validation score
    def cv_score(build_predictor, k=5):
        """Estimate test error via k-fold CV on training data."""
        if elapsed() > deadline - 5:
            return np.inf
        try:
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            folds = np.array_split(indices, k)
            errors = []
            for i in range(k):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                try:
                    pred_func = build_predictor(X[tr_idx], y[tr_idx])
                    if pred_func is None:
                        return np.inf
                    pred_val = pred_func(X[val_idx])
                    pred_val = np.asarray(pred_val, dtype=np.float64)
                    if not np.all(np.isfinite(pred_val)):
                        return np.inf
                    mse = np.mean((pred_val - y[val_idx]) ** 2)
                    errors.append(mse)
                except Exception:
                    return np.inf
            return np.mean(errors)
        except Exception:
            return np.inf

    best_cv = np.inf
    best_predictor = None
    best_model_name = ""

    def try_model(build_predictor_full, name=""):
        """Try a model builder that takes (X, y) and returns a predictor."""
        nonlocal best_cv, best_predictor, best_model_name
        if remaining() < 3:
            return
        try:
            cv = cv_score(build_predictor_full)
            if cv < best_cv:
                best_cv = cv
                # Rebuild with all data
                pred_func = build_predictor_full(X, y)
                if pred_func is not None:
                    # Verify it works on full training set
                    pred_check = pred_func(X)
                    pred_check = np.asarray(pred_check, dtype=np.float64)
                    if np.all(np.isfinite(pred_check)):
                        best_predictor = pred_func
                        best_model_name = name
        except Exception:
            pass

    # ===== Model builders =====

    # 1) Polynomial of degree d
    def build_poly(deg):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                A = np.vander(x_fit, deg + 1, increasing=True)
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    Ai = np.vander(xi, deg + 1, increasing=True)
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 2) Fourier series
    def build_fourier(max_k):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                n_fit = len(x_fit)
                cols = [np.ones(n_fit)]
                for k in range(1, max_k + 1):
                    cols.append(np.sin(k * x_fit))
                    cols.append(np.cos(k * x_fit))
                A = np.column_stack(cols)
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    cols_i = [np.ones(len(xi))]
                    for k in range(1, max_k + 1):
                        cols_i.append(np.sin(k * xi))
                        cols_i.append(np.cos(k * xi))
                    Ai = np.column_stack(cols_i)
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 3) Transformed features: f(x) with linear fit
    def build_transformed(transform_func, name):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                Xf = transform_func(x_fit)
                if not np.all(np.isfinite(Xf)):
                    return None
                # Fit linear in transformed space: a + b*f(x) + c*f(x)^2
                A = np.column_stack([np.ones(len(x_fit)), Xf, Xf ** 2])
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    xf = transform_func(xi)
                    if not np.all(np.isfinite(xf)):
                        return np.zeros(len(xi))
                    Ai = np.column_stack([np.ones(len(xi)), xf, xf ** 2])
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 4) RBF network
    def build_rbf(n_centers, bandwidth):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                centers = np.quantile(x_fit, np.linspace(0.05, 0.95, n_centers))
                Xc = x_fit.reshape(-1, 1)
                C = centers.reshape(1, -1)
                dist2 = (Xc - C) ** 2
                A = np.exp(-dist2 / (2 * bandwidth ** 2))
                A = np.column_stack([np.ones(len(x_fit)), A])
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0].reshape(-1, 1)
                    dist2_i = (xi - C) ** 2
                    Ai = np.exp(-dist2_i / (2 * bandwidth ** 2))
                    Ai = np.column_stack([np.ones(len(xi)), Ai])
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 5) Piecewise linear (spline)
    def build_spline(n_knots):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                knots = np.quantile(x_fit, np.linspace(0, 1, n_knots + 1))
                cols = [np.ones(len(x_fit)), x_fit]
                for knot in knots[1:-1]:
                    cols.append(np.maximum(0, x_fit - knot))
                A = np.column_stack(cols)
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    cols_i = [np.ones(len(xi)), xi]
                    for knot in knots[1:-1]:
                        cols_i.append(np.maximum(0, xi - knot))
                    Ai = np.column_stack(cols_i)
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 5b) Cubic spline via scipy
    def build_cubic_spline(n_knots):
        from scipy.interpolate import CubicSpline
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                y_fit_flat = y_fit.ravel()
                order = np.argsort(x_fit)
                xs = x_fit[order]
                ys = y_fit_flat[order]
                cs = CubicSpline(xs, ys)
                def predict(X_in):
                    xi = X_in[:, 0]
                    return cs(xi)
                return predict
            except Exception:
                return None
        return build

    # 5c) Piecewise polynomial interpolation with adaptive knot placement
    def build_piecewise_poly(n_pieces, poly_deg):
        """Fit piecewise polynomial with knots at quantiles, using RANSAC-like outlier rejection."""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                y_fit_flat = y_fit.ravel()
                n_pts = len(x_fit)
                
                # Sort by x
                order = np.argsort(x_fit)
                xs = x_fit[order]
                ys = y_fit_flat[order]
                
                # Adaptive knot placement: use quantiles but adjust based on local curvature
                knot_quantiles = np.linspace(0, 1, n_pieces + 1)
                knots = np.quantile(xs, knot_quantiles)
                # Ensure unique knots
                knots = np.unique(kots) if len(knots) > 1 else knots
                if len(knots) < n_pieces + 1:
                    knots = np.linspace(xs[0], xs[-1], n_pieces + 1)
                
                # Assign each point to a piece
                piece_idx = np.searchsorted(knots, xs, side='right') - 1
                piece_idx = np.clip(piece_idx, 0, n_pieces - 1)
                
                # Fit piecewise polynomial
                # Build design matrix
                cols_list = []
                for p in range(n_pieces):
                    mask_p = (piece_idx == p)
                    # Local polynomial basis: (x - knot_p)^j for j=0..poly_deg
                    local_x = xs[mask_p] - knots[p]
                    for j in range(poly_deg + 1):
                        col = np.zeros(n_pts)
                        col[mask_p] = local_x ** j
                        cols_list.append(col)
                
                A = np.column_stack(cols_list)
                w, *_ = np.linalg.lstsq(A, ys, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                
                def predict(X_in):
                    xi = X_in[:, 0]
                    n_pred = len(xi)
                    pred = np.zeros(n_pred)
                    for p in range(n_pieces):
                        # Points in this piece: xi >= knots[p] and xi < knots[p+1]
                        if p == n_pieces - 1:
                            mask_p = xi >= knots[p]
                        else:
                            mask_p = (xi >= knots[p]) & (xi < knots[p + 1])
                        if np.any(mask_p):
                            local_x = xi[mask_p] - knots[p]
                            for j in range(poly_deg + 1):
                                pred[mask_p] += w[p * (poly_deg + 1) + j] * (local_x ** j)
                    return pred
                return predict
            except Exception:
                return None
        return build

    # 5d) High-order piecewise polynomial with outlier rejection
    def build_piecewise_poly_robust(n_pieces, poly_deg, n_iter_ransac=5):
        """Piecewise polynomial with simple outlier rejection."""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                y_fit_flat = y_fit.ravel()
                n_pts = len(x_fit)
                
                order = np.argsort(x_fit)
                xs = x_fit[order]
                ys = y_fit_flat[order]
                
                # Initial fit
                knot_quantiles = np.linspace(0, 1, n_pieces + 1)
                knots = np.quantile(xs, knot_quantiles)
                knots = np.unique(knots)
                if len(knots) < n_pieces + 1:
                    knots = np.linspace(xs[0], xs[-1], n_pieces + 1)
                
                piece_idx = np.searchsorted(knots, xs, side='right') - 1
                piece_idx = np.clip(piece_idx, 0, n_pieces - 1)
                
                def build_design(x_sorted, knots_arr, piece_idx_arr, poly_deg_arr):
                    n_p = len(knots_arr) - 1
                    cols_list = []
                    for p in range(n_p):
                        mask_p = (piece_idx_arr == p)
                        local_x = x_sorted[mask_p] - knots_arr[p]
                        for j in range(poly_deg_arr + 1):
                            col = np.zeros(len(x_sorted))
                            col[mask_p] = local_x ** j
                            cols_list.append(col)
                    return np.column_stack(cols_list)
                
                # Outlier rejection iterations
                inlier_mask = np.ones(n_pts, dtype=bool)
                for _ in range(n_iter_ransac):
                    A = build_design(xs[inlier_mask], knots, piece_idx[inlier_mask], poly_deg)
                    if A.shape[0] < A.shape[1]:
                        break
                    w, *_ = np.linalg.lstsq(A, ys[inlier_mask], rcond=None)
                    if not np.all(np.isfinite(w)):
                        break
                    # Compute residuals on all points
                    A_all = build_design(xs, knots, piece_idx, poly_deg)
                    pred_all = A_all @ w
                    residuals = np.abs(ys - pred_all)
                    # Reject outliers: top 5% by residual
                    if np.median(residuals) > 0:
                        threshold = np.percentile(residuals, 95)
                        inlier_mask = residuals <= threshold
                        if np.sum(inlier_mask) < A.shape[1] + 1:
                            inlier_mask = residuals <= np.percentile(residuals, 99)
                
                A = build_design(xs[inlier_mask], knots, piece_idx[inlier_mask], poly_deg)
                w, *_ = np.linalg.lstsq(A, ys[inlier_mask], rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                
                def predict(X_in):
                    xi = X_in[:, 0]
                    n_pred = len(xi)
                    pred = np.zeros(n_pred)
                    n_p = len(knots) - 1
                    for p in range(n_p):
                        if p == n_p - 1:
                            mask_p = xi >= knots[p]
                        else:
                            mask_p = (xi >= knots[p]) & (xi < knots[p + 1])
                        if np.any(mask_p):
                            local_x = xi[mask_p] - knots[p]
                            for j in range(poly_deg + 1):
                                pred[mask_p] += w[p * (poly_deg + 1) + j] * (local_x ** j)
                    return pred
                return predict
            except Exception:
                return None
        return build

    # 6) Exponential / log / reciprocal forms
    def build_exp_form(a_range, b_range):
        """Fit y = a * exp(b*x) + c"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                # Nonlinear fit: y = a*exp(b*x) + c
                def model(x, a, b, c):
                    return a * np.exp(np.clip(b * x, -20, 20)) + c
                # Initial guess
                y_mean = np.mean(y_fit)
                y_range = np.max(y_fit) - np.min(y_fit)
                p0 = [y_range / 2, 0.1, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=1000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c = popt
                    def predict(X_in):
                        xi = X_in[:, 0]
                        return a * np.exp(np.clip(b * xi, -20, 20)) + c
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    def build_log_form():
        """Fit y = a*log(x+b) + c for x>0"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                # Only valid if x > 0 mostly
                if np.min(x_fit) < 0:
                    return None
                def model(x, a, b, c):
                    arg = x + b
                    if np.min(arg) <= 0:
                        return np.full(len(x), np.nan)
                    return a * np.log(arg) + c
                y_mean = np.mean(y_fit)
                p0 = [1.0, 0.1, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=1000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c = popt
                    def predict(X_in):
                        xi = X_in[:, 0]
                        arg = xi + b
                        mask = arg > 0
                        res = np.zeros(len(xi))
                        res[mask] = a * np.log(arg[mask]) + c
                        if not np.all(mask):
                            res[~mask] = a * np.log(max(b + 0.1, 0.1)) + c
                        return res
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    def build_reciprocal_form():
        """Fit y = a/(x+b) + c"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                def model(x, a, b, c):
                    denom = x + b
                    mask = np.abs(denom) > 1e-10
                    res = np.full(len(x), c)
                    res[mask] = a / denom[mask] + c
                    return res
                y_mean = np.mean(y_fit)
                p0 = [1.0, 0.5, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=1000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c = popt
                    def predict(X_in):
                        xi = X_in[:, 0]
                        denom = xi + b
                        mask = np.abs(denom) > 1e-10
                        res = np.full(len(xi), c)
                        res[mask] = a / denom[mask] + c
                        return res
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    def build_quadratic_form():
        """Fit y = a*x^2 + b*x + c (same as poly degree 2, but explicit)"""
        return build_poly(2)

    def build_cubic_form():
        return build_poly(3)

    def build_sin_cos_single():
        """y = a*sin(x) + b*cos(x) + c"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                A = np.column_stack([np.ones(len(x_fit)), np.sin(x_fit), np.cos(x_fit)])
                w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                if not np.all(np.isfinite(w)):
                    return None
                def predict(X_in):
                    xi = X_in[:, 0]
                    Ai = np.column_stack([np.ones(len(xi)), np.sin(xi), np.cos(xi)])
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    def build_tanh_form():
        """y = a*tanh(b*x) + c"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                def model(x, a, b, c):
                    return a * np.tanh(b * x) + c
                y_range = np.max(y_fit) - np.min(y_fit)
                y_mean = np.mean(y_fit)
                p0 = [y_range / 2, 1.0, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=1000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c = popt
                    def predict(X_in):
                        xi = X_in[:, 0]
                        return a * np.tanh(b * xi) + c
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    def build_gaussian_bump():
        """y = a * exp(-((x-b)/c)^2) + d"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                def model(x, a, b, c, d):
                    return a * np.exp(-((x - b) / c) ** 2) + d
                y_range = np.max(y_fit) - np.min(y_fit)
                y_mean = np.mean(y_fit)
                x_mean = np.mean(x_fit)
                p0 = [y_range, x_mean, 1.0, y_mean - y_range / 2]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=2000)
                    if not np.all(np.isfinite(popt)):
                        return None
                    a, b, c, d = popt
                    if abs(c) < 1e-10:
                        return None
                    def predict(X_in):
                        xi = X_in[:, 0]
                        return a * np.exp(-((xi - b) / c) ** 2) + d
                    return predict
                except Exception:
                    return None
            except Exception:
                return None
        return build

    # 7) Adaptive basis expansion with Ridge Regression
    def build_adaptive_basis(n_features=20):
        """Adaptively select basis functions and fit with Ridge Regression."""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                # Generate candidate basis functions
                basis_candidates = []
                basis_names = []
                
                # Powers
                for p in range(1, 8):
                    basis_candidates.append(x_fit ** p)
                    basis_names.append(f"x^{p}")
                
                # Trigonometric
                for k in range(1, 6):
                    basis_candidates.append(np.sin(k * x_fit))
                    basis_candidates.append(np.cos(k * x_fit))
                    basis_names.append(f"sin({k}x)")
                    basis_names.append(f"cos({k}x)")
                
                # Exponential
                basis_candidates.append(np.exp(np.clip(x_fit, -10, 10)))
                basis_names.append("exp(x)")
                
                # Hyperbolic
                basis_candidates.append(np.tanh(x_fit))
                basis_names.append("tanh(x)")
                
                # Reciprocal
                safe_x = np.where(np.abs(x_fit) < 1e-10, 1e-10, x_fit)
                basis_candidates.append(1.0 / safe_x)
                basis_names.append("1/x")
                
                # Log
                basis_candidates.append(np.log(np.abs(x_fit) + 1e-10))
                basis_names.append("log|x|")
                
                # Sqrt
                basis_candidates.append(np.sqrt(np.abs(x_fit)))
                basis_names.append("sqrt|x|")
                
                # Inverse quadratic
                basis_candidates.append(1.0 / (1.0 + x_fit ** 2))
                basis_names.append("1/(1+x^2)")
                
                # Select top n_features by correlation with y
                correlations = []
                for i, basis in enumerate(basis_candidates):
                    if not np.all(np.isfinite(basis)):
                        correlations.append(0)
                        continue
                    corr = np.corrcoef(basis, y_fit)[0, 1]
                    correlations.append(abs(corr) if np.isfinite(corr) else 0)
                
                correlations = np.array(correlations)
                top_indices = np.argsort(correlations)[-n_features:]
                
                selected_basis = [basis_candidates[i] for i in top_indices]
                selected_names = [basis_names[i] for i in top_indices]
                
                # Fit with Ridge Regression
                A = np.column_stack([np.ones(len(x_fit))] + selected_basis)
                
                # Use Ridge with lambda selected via CV
                best_lambda = 1e-3
                best_cv_ridge = np.inf
                for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                    try:
                        # Simple CV for lambda
                        indices = np.arange(len(x_fit))
                        np.random.seed(42)
                        np.random.shuffle(indices)
                        folds = np.array_split(indices, 5)
                        errors = []
                        for i in range(5):
                            val_idx = folds[i]
                            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                            A_tr = A[tr_idx]
                            y_tr = y_fit[tr_idx]
                            w = np.linalg.solve(A_tr.T @ A_tr + lam * np.eye(A_tr.shape[1]), A_tr.T @ y_tr)
                            pred_val = A[val_idx] @ w
                            mse = np.mean((pred_val - y_fit[val_idx]) ** 2)
                            errors.append(mse)
                        cv_ridge = np.mean(errors)
                        if cv_ridge < best_cv_ridge:
                            best_cv_ridge = cv_ridge
                            best_lambda = lam
                    except Exception:
                        continue
                
                # Fit with best lambda
                w = np.linalg.solve(A.T @ A + best_lambda * np.eye(A.shape[1]), A.T @ y_fit)
                if not np.all(np.isfinite(w)):
                    return None
                
                def predict(X_in):
                    xi = X_in[:, 0]
                    safe_xi = np.where(np.abs(xi) < 1e-10, 1e-10, xi)
                    cols = [np.ones(len(xi))]
                    for name in selected_names:
                        if name.startswith("x^"):
                            p = int(name[2:])
                            cols.append(xi ** p)
                        elif name.startswith("sin("):
                            k = int(name[4:-1])
                            cols.append(np.sin(k * xi))
                        elif name.startswith("cos("):
                            k = int(name[4:-1])
                            cols.append(np.cos(k * xi))
                        elif name == "exp(x)":
                            cols.append(np.exp(np.clip(xi, -10, 10)))
                        elif name == "tanh(x)":
                            cols.append(np.tanh(xi))
                        elif name == "1/x":
                            cols.append(1.0 / safe_xi)
                        elif name == "log|x|":
                            cols.append(np.log(np.abs(xi) + 1e-10))
                        elif name == "sqrt|x|":
                            cols.append(np.sqrt(np.abs(xi)))
                        elif name == "1/(1+x^2)":
                            cols.append(1.0 / (1.0 + xi ** 2))
                    Ai = np.column_stack(cols)
                    return Ai @ w
                return predict
            except Exception:
                return None
        return build

    # 8) High-precision polish for parametric models
    
    def build_polished_exp():
        """Fit y = a * exp(b*x) + c with high precision."""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                def model(x, a, b, c):
                    return a * np.exp(np.clip(b * x, -20, 20)) + c
                
                # Initial guess via linearization
                y_mean = np.mean(y_fit)
                y_range = np.max(y_fit) - np.min(y_fit)
                x_mean = np.mean(x_fit)
                
                # Try multiple initial guesses
                best_params = None
                best_mse = np.inf
                for a0 in [-y_range/2, y_range/2, y_range]:
                    for b0 in [-1.0, -0.5, 0.1, 0.5, 1.0]:
                        for c0 in [y_mean - y_range/2, y_mean, y_mean + y_range/2]:
                            p0 = [a0, b0, c0]
                            try:
                                popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=5000)
                                if np.all(np.isfinite(popt)):
                                    pred = model(x_fit, *popt)
                                    mse = np.mean((pred - y_fit) ** 2)
                                    if mse < best_mse:
                                        best_mse = mse
                                        best_params = popt
                            except Exception:
                                continue
                
                if best_params is None:
                    return None
                
                a, b, c = best_params
                def predict(X_in):
                    xi = X_in[:, 0]
                    return a * np.exp(np.clip(b * xi, -20, 20)) + c
                return predict
            except Exception:
                return None
        return build

    def build_polished_log():
        """Fit y = a*log(x+b) + c with high precision for x>0."""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                if np.min(x_fit) < 0:
                    return None
                
                def model(x, a, b, c):
                    arg = x + b
                    if np.min(arg) <= 0:
                        return np.full(len(x), np.nan)
                    return a * np.log(arg) + c
                
                y_mean = np.mean(y_fit)
                best_params = None
                best_mse = np.inf
                for a0 in [-1.0, 1.0, 2.0]:
                    for b0 in [0.1, 0.5, 1.0]:
                        for c0 in [y_mean - 1.0, y_mean, y_mean + 1.0]:
                            p0 = [a0, b0, c0]
                            try:
                                popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=5000)
                                if np.all(np.isfinite(popt)):
                                    pred = model(x_fit, *popt)
                                    if np.all(np.isfinite(pred)):
                                        mse = np.mean((pred - y_fit) ** 2)
                                        if mse < best_mse:
                                            best_mse = mse
                                            best_params = popt
                            except Exception:
                                continue
                
                if best_params is None:
                    return None
                
                a, b, c = best_params
                def predict(X_in):
                    xi = X_in[:, 0]
                    arg = xi + b
                    mask = arg > 0
                    res = np.zeros(len(xi))
                    res[mask] = a * np.log(arg[mask]) + c
                    if not np.all(mask):
                        res[~mask] = a * np.log(max(b + 0.1, 0.1)) + c
                    return res
                return predict
            except Exception:
                return None
        return build

    # ===== Try all models =====

    # Polynomials
    for deg in range(1, 13):
        if remaining() < 5:
            break
        try_model(build_poly(deg), f"poly_{deg}")

    # Fourier
    for max_k in range(1, 10):
        if remaining() < 5:
            break
        try_model(build_fourier(max_k), f"fourier_{max_k}")

    # Transformed features
    transforms = [
        ("identity", lambda t: t),
        ("reciprocal", lambda t: 1.0 / np.where(np.abs(t) < 1e-10, 1e-10, t)),
        ("inv_quad", lambda t: 1.0 / (1.0 + t ** 2)),
        ("exp", lambda t: np.exp(np.clip(t, -10, 10))),
        ("tanh", lambda t: np.tanh(t)),
        ("log_abs", lambda t: np.log(np.abs(t) + 1e-10)),
        ("sqrt_abs", lambda t: np.sqrt(np.abs(t))),
        ("x_cubed", lambda t: t ** 3),
    ]
    for name, tf in transforms:
        if remaining() < 5:
            break
        try_model(build_transformed(tf, name), f"transform_{name}")

    # RBF
    for nc in [5, 10, 20, 30, 50]:
        for bw in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
            if remaining() < 5:
                break
            try_model(build_rbf(nc, bw), f"rbf_{nc}_{bw}")

    # Splines
    for nk in [5, 10, 20, 30, 50]:
        if remaining() < 5:
            break
        try_model(build_spline(nk), f"spline_{nk}")
    
    # Cubic splines
    for nk in [10, 20, 50, 100]:
        if remaining() < 5:
            break
        try_model(build_cubic_spline(nk), f"cubic_spline_{nk}")
    
    # Piecewise polynomial with adaptive knots
    for npieces in [10, 20, 40]:
        for pdeg in [2, 3, 4]:
            if remaining() < 5:
                break
            try_model(build_piecewise_poly(npieces, pdeg), f"pw_poly_{npieces}_{pdeg}")
    
    # Robust piecewise polynomial with outlier rejection
    for npieces in [10, 20]:
        for pdeg in [3, 4]:
            if remaining() < 5:
                break
            try_model(build_piecewise_poly_robust(npieces, pdeg), f"pw_poly_robust_{npieces}_{pdeg}")

    # Parametric forms
    if remaining() > 10:
        try_model(build_exp_form(None, None), "exp_form")
    if remaining() > 10:
        try_model(build_log_form(), "log_form")
    if remaining() > 10:
        try_model(build_reciprocal_form(), "reciprocal_form")
    if remaining() > 10:
        try_model(build_cubic_form(), "cubic_form")
    if remaining() > 10:
        try_model(build_sin_cos_single(), "sin_cos_single")
    if remaining() > 10:
        try_model(build_tanh_form(), "tanh_form")
    if remaining() > 10:
        try_model(build_gaussian_bump(), "gaussian_bump")

    # Adaptive basis expansion
    if remaining() > 15:
        try_model(build_adaptive_basis(20), "adaptive_basis_20")
    if remaining() > 15:
        try_model(build_adaptive_basis(30), "adaptive_basis_30")
    if remaining() > 15:
        try_model(build_adaptive_basis(40), "adaptive_basis_40")

    # High-precision polish
    if remaining() > 10:
        try_model(build_polished_exp(), "polished_exp")
    if remaining() > 10:
        try_model(build_polished_log(), "polished_log")

    # Additional: try combinations of basis functions
    if remaining() > 10:
        # x, x^2, sin(x), cos(x), exp(x), 1/x
        def build_combo():
            def build(X_fit, y_fit):
                try:
                    x_fit = X_fit[:, 0]
                    safe_x = np.where(np.abs(x_fit) < 1e-10, 1e-10, x_fit)
                    cols = [
                        np.ones(len(x_fit)),
                        x_fit,
                        x_fit ** 2,
                        np.sin(x_fit),
                        np.cos(x_fit),
                        np.exp(np.clip(x_fit, -10, 10)),
                        1.0 / safe_x,
                        np.tanh(x_fit),
                    ]
                    A = np.column_stack(cols)
                    w, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
                    if not np.all(np.isfinite(w)):
                        return None
                    def predict(X_in):
                        xi = X_in[:, 0]
                        safe_xi = np.where(np.abs(xi) < 1e-10, 1e-10, xi)
                        cols_i = [
                            np.ones(len(xi)),
                            xi,
                            xi ** 2,
                            np.sin(xi),
                            np.cos(xi),
                            np.exp(np.clip(xi, -10, 10)),
                            1.0 / safe_xi,
                            np.tanh(xi),
                        ]
                        Ai = np.column_stack(cols_i)
                        return Ai @ w
                    return predict
                except Exception:
                    return None
            return build
        try_model(build_combo(), "combo")

    # Additional: try higher-degree polynomials and Fourier with more terms
    for deg in range(13, 21):
        if remaining() < 3:
            break
        try_model(build_poly(deg), f"poly_{deg}")
    
    for max_k in range(10, 20):
        if remaining() < 3:
            break
        try_model(build_fourier(max_k), f"fourier_{max_k}")
    
    # Fallback: constant
    if best_predictor is None:
        mean_y = np.mean(y)
        best_predictor = lambda X_in: np.full(len(X_in), mean_y)

    # Final safety wrapper
    def final_predict(X_in):
        X_in = np.asarray(X_in, dtype=np.float64)
        try:
            res = best_predictor(X_in)
            res = np.asarray(res, dtype=np.float64)
            if res.shape != (len(X_in),):
                res = np.atleast_1d(res).flatten()
                if len(res) != len(X_in):
                    res = np.resize(res, len(X_in))
            if not np.all(np.isfinite(res)):
                res = np.where(np.isfinite(res), res, np.mean(y))
            return res
        except Exception:
            return np.full(len(X_in), np.mean(y))

    return final_predict

# EVOLVE-BLOCK-END
