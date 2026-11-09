"""Symbolic regression via piecewise interpolation with adaptive refinement."""
import time
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline

# EVOLVE-BLOCK-START
import numpy as np

def fit(X_train, y_train):
    """
    Fit a regression model using piecewise interpolation and diverse model search.
    Uses cross-validation to select the best model, with cubic spline as the
    primary candidate for smooth noiseless functions.
    """
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0]
    x_min, x_max = np.min(x), np.max(x)

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

    def train_mse(predict_func):
        """Compute MSE on the full training set."""
        try:
            pred = predict_func(X)
            pred = np.asarray(pred, dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                return np.inf
            return np.mean((pred - y) ** 2)
        except Exception:
            return np.inf

    best_score = np.inf
    best_predictor = None
    best_model_name = ""

    def try_model(build_predictor_full, name="", skip_cv=False):
        """Try a model builder that takes (X, y) and returns a predictor."""
        nonlocal best_score, best_predictor, best_model_name
        if remaining() < 3:
            return
        try:
            # Build on full data first
            pred_func = build_predictor_full(X, y)
            if pred_func is None:
                return
            # Verify it works on full training set
            pred_check = pred_func(X)
            pred_check = np.asarray(pred_check, dtype=np.float64)
            if not np.all(np.isfinite(pred_check)):
                return
            train_err = np.mean((pred_check - y) ** 2)
            
            if skip_cv:
                # For interpolation models, use training error as proxy
                score = train_err
            else:
                # Use CV for parametric models
                cv = cv_score(build_predictor_full)
                if not np.isfinite(cv):
                    return
                score = cv
            
            if score < best_score:
                best_score = score
                best_predictor = pred_func
                best_model_name = name
        except Exception:
            pass

    # ===== Model builders =====

    # 1) Cubic spline interpolation (primary candidate for smooth noiseless data)
    def build_cubic_spline():
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                y_fit_flat = y_fit.ravel()
                order = np.argsort(x_fit)
                xs = x_fit[order]
                ys = y_fit_flat[order]
                # Check for duplicate x values
                if len(np.unique(xs)) < len(xs):
                    # Average duplicates
                    unique_x, inverse = np.unique(xs, return_inverse=True)
                    ys_avg = np.zeros(len(unique_x))
                    counts = np.zeros(len(unique_x))
                    np.add.at(ys_avg, inverse, ys)
                    np.add.at(counts, inverse, 1)
                    ys_avg /= counts
                    xs, ys = unique_x, ys_avg
                cs = CubicSpline(xs, ys)
                x_lo, x_hi = xs[0], xs[-1]
                y_lo, y_hi = np.min(ys), np.max(ys)
                y_range = y_hi - y_lo if y_hi > y_lo else 1.0
                def predict(X_in):
                    xi = X_in[:, 0]
                    # Clamp to training range to avoid wild extrapolation
                    xi_clamped = np.clip(xi, x_lo, x_hi)
                    res = cs(xi_clamped)
                    # For points outside training range, use linear extrapolation from endpoints
                    outside_left = xi < x_lo
                    outside_right = xi > x_hi
                    if np.any(outside_left) or np.any(outside_right):
                        # Linear extrapolation using first/last two points
                        if len(xs) >= 2:
                            slope_left = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
                            slope_right = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0.0
                            if np.any(outside_left):
                                res[outside_left] = ys[0] + slope_left * (xi[outside_left] - xs[0])
                            if np.any(outside_right):
                                res[outside_right] = ys[-1] + slope_right * (xi[outside_right] - xs[-1])
                        # Clamp to reasonable range
                        res = np.clip(res, y_lo - 2 * y_range, y_hi + 2 * y_range)
                    return res
                return predict
            except Exception:
                return None
        return build

    # 2) PCHIP interpolation (preserves shape, no overshoot)
    def build_pchip():
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                y_fit_flat = y_fit.ravel()
                order = np.argsort(x_fit)
                xs = x_fit[order]
                ys = y_fit_flat[order]
                if len(np.unique(xs)) < len(xs):
                    unique_x, inverse = np.unique(xs, return_inverse=True)
                    ys_avg = np.zeros(len(unique_x))
                    counts = np.zeros(len(unique_x))
                    np.add.at(ys_avg, inverse, ys)
                    np.add.at(counts, inverse, 1)
                    ys_avg /= counts
                    xs, ys = unique_x, ys_avg
                pchip = PchipInterpolator(xs, ys)
                x_lo, x_hi = xs[0], xs[-1]
                y_lo, y_hi = np.min(ys), np.max(ys)
                y_range = y_hi - y_lo if y_hi > y_lo else 1.0
                def predict(X_in):
                    xi = X_in[:, 0]
                    xi_clamped = np.clip(xi, x_lo, x_hi)
                    res = pchip(xi_clamped)
                    outside_left = xi < x_lo
                    outside_right = xi > x_hi
                    if np.any(outside_left) or np.any(outside_right):
                        if len(xs) >= 2:
                            slope_left = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
                            slope_right = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0.0
                            if np.any(outside_left):
                                res[outside_left] = ys[0] + slope_left * (xi[outside_left] - xs[0])
                            if np.any(outside_right):
                                res[outside_right] = ys[-1] + slope_right * (xi[outside_right] - xs[-1])
                        res = np.clip(res, y_lo - 2 * y_range, y_hi + 2 * y_range)
                    return res
                return predict
            except Exception:
                return None
        return build

    # 3) High-order spline with smoothing
    def build_smooth_spline(s=1.0):
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                y_fit_flat = y_fit.ravel()
                order = np.argsort(x_fit)
                xs = x_fit[order]
                ys = y_fit_flat[order]
                if len(np.unique(xs)) < len(xs):
                    unique_x, inverse = np.unique(xs, return_inverse=True)
                    ys_avg = np.zeros(len(unique_x))
                    counts = np.zeros(len(unique_x))
                    np.add.at(ys_avg, inverse, ys)
                    np.add.at(counts, inverse, 1)
                    ys_avg /= counts
                    xs, ys = unique_x, ys_avg
                spl = make_interp_spline(xs, ys, k=3, s=s)
                x_lo, x_hi = xs[0], xs[-1]
                y_lo, y_hi = np.min(ys), np.max(ys)
                y_range = y_hi - y_lo if y_hi > y_lo else 1.0
                def predict(X_in):
                    xi = X_in[:, 0]
                    xi_clamped = np.clip(xi, x_lo, x_hi)
                    res = spl(xi_clamped)
                    outside_left = xi < x_lo
                    outside_right = xi > x_hi
                    if np.any(outside_left) or np.any(outside_right):
                        if len(xs) >= 2:
                            slope_left = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
                            slope_right = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0.0
                            if np.any(outside_left):
                                res[outside_left] = ys[0] + slope_left * (xi[outside_left] - xs[0])
                            if np.any(outside_right):
                                res[outside_right] = ys[-1] + slope_right * (xi[outside_right] - xs[-1])
                        res = np.clip(res, y_lo - 2 * y_range, y_hi + 2 * y_range)
                    return res
                return predict
            except Exception:
                return None
        return build

    # 4) Polynomial of degree d
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

    # 5) Fourier series
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

    # 6) RBF network
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

    # 7) Piecewise polynomial with adaptive knots
    def build_piecewise_poly(n_pieces, poly_deg):
        """Fit piecewise polynomial with knots at quantiles."""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                y_fit_flat = y_fit.ravel()
                n_pts = len(x_fit)
                
                # Sort by x
                order = np.argsort(x_fit)
                xs = x_fit[order]
                ys = y_fit_flat[order]
                
                # Handle duplicates
                if len(np.unique(xs)) < n_pts:
                    unique_x, inverse = np.unique(xs, return_inverse=True)
                    ys_avg = np.zeros(len(unique_x))
                    counts = np.zeros(len(unique_x))
                    np.add.at(ys_avg, inverse, ys)
                    np.add.at(counts, inverse, 1)
                    ys_avg /= counts
                    xs, ys = unique_x, ys_avg
                    n_pts = len(xs)
                
                # Adaptive knot placement
                knot_quantiles = np.linspace(0, 1, n_pieces + 1)
                knots = np.quantile(xs, knot_quantiles)
                knots = np.unique(knots)
                if len(knots) < n_pieces + 1:
                    knots = np.linspace(xs[0], xs[-1], n_pieces + 1)
                
                # Assign each point to a piece
                piece_idx = np.searchsorted(knots, xs, side='right') - 1
                piece_idx = np.clip(piece_idx, 0, n_pieces - 1)
                
                # Fit piecewise polynomial
                cols_list = []
                for p in range(n_pieces):
                    mask_p = (piece_idx == p)
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

    # 8) Exponential form
    def build_exp_form():
        """Fit y = a * exp(b*x) + c"""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                def model(x, a, b, c):
                    return a * np.exp(np.clip(b * x, -20, 20)) + c
                y_mean = np.mean(y_fit)
                y_range = np.max(y_fit) - np.min(y_fit)
                p0 = [y_range / 2, 0.1, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=2000)
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

    # 9) Log form
    def build_log_form():
        """Fit y = a*log(x+b) + c for x>0"""
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
                p0 = [1.0, 0.1, y_mean]
                try:
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=2000)
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

    # 10) Tanh form
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
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=2000)
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

    # 11) Gaussian bump
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
                    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=3000)
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

    # 12) Adaptive basis with Ridge
    def build_adaptive_basis(n_features=20):
        """Adaptively select basis functions and fit with Ridge Regression."""
        def build(X_fit, y_fit):
            try:
                x_fit = X_fit[:, 0]
                basis_candidates = []
                basis_names = []
                
                for p in range(1, 8):
                    basis_candidates.append(x_fit ** p)
                    basis_names.append(f"x^{p}")
                
                for k in range(1, 6):
                    basis_candidates.append(np.sin(k * x_fit))
                    basis_candidates.append(np.cos(k * x_fit))
                    basis_names.append(f"sin({k}x)")
                    basis_names.append(f"cos({k}x)")
                
                basis_candidates.append(np.exp(np.clip(x_fit, -10, 10)))
                basis_names.append("exp(x)")
                
                basis_candidates.append(np.tanh(x_fit))
                basis_names.append("tanh(x)")
                
                safe_x = np.where(np.abs(x_fit) < 1e-10, 1e-10, x_fit)
                basis_candidates.append(1.0 / safe_x)
                basis_names.append("1/x")
                
                basis_candidates.append(np.log(np.abs(x_fit) + 1e-10))
                basis_names.append("log|x|")
                
                basis_candidates.append(np.sqrt(np.abs(x_fit)))
                basis_names.append("sqrt|x|")
                
                basis_candidates.append(1.0 / (1.0 + x_fit ** 2))
                basis_names.append("1/(1+x^2)")
                
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
                
                A = np.column_stack([np.ones(len(x_fit))] + selected_basis)
                
                best_lambda = 1e-3
                best_cv_ridge = np.inf
                for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                    try:
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

    # ===== Try all models, prioritizing interpolation methods =====

    # Primary: Cubic spline (best for smooth noiseless data)
    try_model(build_cubic_spline(), "cubic_spline")
    
    # Secondary: PCHIP
    if remaining() > 5:
        try_model(build_pchip(), "pchip")
    
    # Smoothed splines
    for s in [0.0, 0.1, 1.0, 10.0]:
        if remaining() < 5:
            break
        try_model(build_smooth_spline(s), f"smooth_spline_{s}")
    
    # Piecewise polynomials
    for npieces in [10, 20, 40, 80]:
        for pdeg in [2, 3, 4, 5]:
            if remaining() < 5:
                break
            try_model(build_piecewise_poly(npieces, pdeg), f"pw_poly_{npieces}_{pdeg}")
    
    # Polynomials
    for deg in range(1, 15):
        if remaining() < 5:
            break
        try_model(build_poly(deg), f"poly_{deg}")
    
    # Fourier
    for max_k in range(1, 15):
        if remaining() < 5:
            break
        try_model(build_fourier(max_k), f"fourier_{max_k}")
    
    # RBF
    for nc in [10, 20, 40, 80]:
        for bw in [0.3, 0.5, 1.0, 2.0]:
            if remaining() < 5:
                break
            try_model(build_rbf(nc, bw), f"rbf_{nc}_{bw}")
    
    # Parametric forms
    if remaining() > 10:
        try_model(build_exp_form(), "exp_form")
    if remaining() > 10:
        try_model(build_log_form(), "log_form")
    if remaining() > 10:
        try_model(build_tanh_form(), "tanh_form")
    if remaining() > 10:
        try_model(build_gaussian_bump(), "gaussian_bump")
    
    # Adaptive basis
    if remaining() > 15:
        try_model(build_adaptive_basis(20), "adaptive_basis_20")
    if remaining() > 15:
        try_model(build_adaptive_basis(30), "adaptive_basis_30")

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
