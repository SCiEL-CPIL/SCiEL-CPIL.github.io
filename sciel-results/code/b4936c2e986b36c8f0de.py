# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, PchipInterpolator

_FIT_DEADLINE = 270.0  # safety margin under 300s
_START = None

def _nse_score(y_pred, y_true, var_y):
    """Normalized squared error."""
    mse = np.mean((y_pred - y_true) ** 2)
    return mse / max(var_y, 1e-12)

def _poly_fit(X, y, degree):
    """Polynomial regression for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    F = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    return w, degree

def _poly_predict_from_w(Xt, w, degree):
    x = Xt[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    F = np.column_stack(cols)
    return F @ w

def _exp_fit(X, y):
    """Fit y = a * exp(b * x) + c using log-transform and optimization."""
    x = X[:, 0]
    # Try: y = a * exp(b*x) + c
    # Use scipy minimize
    def params_to_y(p):
        a, b, c = p
        return a * np.exp(np.clip(b * x, -50, 50)) + c
    
    def loss(p):
        try:
            yp = params_to_y(p)
            if not np.all(np.isfinite(yp)):
                return 1e10
            return np.sum((yp - y) ** 2)
        except:
            return 1e10
    
    # Multiple initial guesses
    best_p = None
    best_loss = 1e10
    y_range = np.ptp(y)
    x_range = np.ptp(x)
    
    for a_init in [y_range, 1.0, -1.0, 0.5*y_range]:
        for b_init in [0.1, 0.5, 1.0, -0.1, -0.5, -1.0, 0.01, 2.0]:
            c_init = np.mean(y) - a_init * np.exp(np.clip(b_init * x, -50, 50)).mean()
            p0 = [a_init, b_init, c_init]
            try:
                res = minimize(loss, p0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_p = res.x
            except:
                pass
    
    if best_p is None:
        return None
    
    a, b, c = best_p
    def predict(Xt):
        xt = Xt[:, 0]
        return a * np.exp(np.clip(b * xt, -50, 50)) + c
    return predict, best_loss / len(y)

def _power_fit(X, y):
    """Fit y = a * x^b + c."""
    x = X[:, 0]
    # Only valid for x > 0 or use |x|
    mask = np.abs(x) > 1e-6
    
    def params_to_y(p):
        a, b, c = p
        return a * np.sign(x) * np.power(np.abs(x), b) + c
    
    def loss(p):
        try:
            yp = params_to_y(p)
            if not np.all(np.isfinite(yp)):
                return 1e10
            return np.sum((yp - y) ** 2)
        except:
            return 1e10
    
    best_p = None
    best_loss = 1e10
    y_range = np.ptp(y)
    
    for a_init in [y_range, 1.0, -1.0]:
        for b_init in [0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0]:
            c_init = np.mean(y)
            p0 = [a_init, b_init, c_init]
            try:
                res = minimize(loss, p0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_p = res.x
            except:
                pass
    
    if best_p is None:
        return None
    
    a, b, c = best_p
    def predict(Xt):
        xt = Xt[:, 0]
        return a * np.sign(xt) * np.power(np.abs(xt), b) + c
    return predict, best_loss / len(y)

def _sin_fit(X, y):
    """Fit y = a * sin(b * x + c) + d."""
    x = X[:, 0]
    
    def params_to_y(p):
        a, b, c, d = p
        return a * np.sin(b * x + c) + d
    
    def loss(p):
        try:
            yp = params_to_y(p)
            if not np.all(np.isfinite(yp)):
                return 1e10
            return np.sum((yp - y) ** 2)
        except:
            return 1e10
    
    best_p = None
    best_loss = 1e10
    y_range = np.ptp(y)
    
    for a_init in [y_range, 1.0, 0.5*y_range]:
        for b_init in [0.5, 1.0, 2.0, 3.0, 0.1, 4.0]:
            for c_init in [0.0, np.pi/2, np.pi]:
                d_init = np.mean(y)
                p0 = [a_init, b_init, c_init, d_init]
                try:
                    res = minimize(loss, p0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_p = res.x
                except:
                    pass
    
    if best_p is None:
        return None
    
    a, b, c, d = best_p
    def predict(Xt):
        xt = Xt[:, 0]
        return a * np.sin(b * xt + c) + d
    return predict, best_loss / len(y)

def _spline_fit(X, y, n_knots=10):
    """Cubic spline interpolation."""
    x = X[:, 0]
    # Sort by x
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    
    # Use Pchip for shape-preserving
    try:
        pchip = PchipInterpolator(x_sorted, y_sorted)
        def predict(Xt):
            xt = Xt[:, 0]
            return pchip(xt)
        # Compute training MSE
        y_pred = pchip(x)
        mse = np.mean((y_pred - y) ** 2)
        return predict, mse
    except:
        return None, 1e10

def _cubic_spline_fit(X, y, n_knots=10):
    """Cubic spline with specified number of knots."""
    x = X[:, 0]
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    
    try:
        # Select knot locations
        if n_knots >= 2:
            knot_indices = np.linspace(0, len(x_sorted)-1, n_knots).astype(int)
            knots = x_sorted[knot_indices]
        else:
            knots = None
        
        cs = CubicSpline(x_sorted, y_sorted, bc_type='natural')
        def predict(Xt):
            xt = Xt[:, 0]
            return cs(xt)
        y_pred = cs(x)
        mse = np.mean((y_pred - y) ** 2)
        return predict, mse
    except:
        return None, 1e10

def _gauss_fit(X, y):
    """Fit y = a * exp(-((x-b)^2)/(2*c^2)) + d."""
    x = X[:, 0]
    
    def params_to_y(p):
        a, b, c, d = p
        c = max(abs(c), 1e-6)
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + d
    
    def loss(p):
        try:
            yp = params_to_y(p)
            if not np.all(np.isfinite(yp)):
                return 1e10
            return np.sum((yp - y) ** 2)
        except:
            return 1e10
    
    best_p = None
    best_loss = 1e10
    y_range = np.ptp(y)
    x_range = np.ptp(x)
    
    for a_init in [y_range, 1.0]:
        for b_init in [x.mean(), x.min(), x.max()]:
            for c_init in [x_range/4, x_range/2, x_range/8]:
                d_init = np.min(y)
                p0 = [a_init, b_init, c_init, d_init]
                try:
                    res = minimize(loss, p0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_p = res.x
                except:
                    pass
    
    if best_p is None:
        return None
    
    a, b, c, d = best_p
    c = max(abs(c), 1e-6)
    def predict(Xt):
        xt = Xt[:, 0]
        return a * np.exp(-((xt - b) ** 2) / (2 * c ** 2)) + d
    return predict, best_loss / len(y)

def fit(X_train, y_train):
    global _START
    if _START is None:
        _START = time.time()
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    var_y = float(np.var(y))
    
    if var_y < 1e-12:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    # Collect candidate predictors with their scores
    candidates = []  # list of (predict_fn, nse_score)
    
    # 1. Polynomial regression (degrees 1-8)
    for deg in range(1, 9):
        try:
            w, degree = _poly_fit(X, y, deg)
            y_pred = _poly_predict_from_w(X, w, deg)
            if np.all(np.isfinite(y_pred)):
                score = _nse_score(y_pred, y, var_y)
                candidates.append((lambda Xt, w=w, d=deg: _poly_predict_from_w(Xt, w, d), score))
        except:
            pass
    
    # 2. Exponential fit
    result = _exp_fit(X, y)
    if result is not None:
        pred_fn, mse = result
        y_pred = pred_fn(X)
        if np.all(np.isfinite(y_pred)):
            score = _nse_score(y_pred, y, var_y)
            candidates.append((pred_fn, score))
    
    # 3. Power law fit
    result = _power_fit(X, y)
    if result is not None:
        pred_fn, mse = result
        y_pred = pred_fn(X)
        if np.all(np.isfinite(y_pred)):
            score = _nse_score(y_pred, y, var_y)
            candidates.append((pred_fn, score))
    
    # 4. Sinusoidal fit
    result = _sin_fit(X, y)
    if result is not None:
        pred_fn, mse = result
        y_pred = pred_fn(X)
        if np.all(np.isfinite(y_pred)):
            score = _nse_score(y_pred, y, var_y)
            candidates.append((pred_fn, score))
    
    # 5. Spline fits
    result = _spline_fit(X, y)
    if result[0] is not None:
        pred_fn, mse = result
        y_pred = pred_fn(X)
        if np.all(np.isfinite(y_pred)):
            score = _nse_score(y_pred, y, var_y)
            candidates.append((pred_fn, score))
    
    result = _cubic_spline_fit(X, y)
    if result[0] is not None:
        pred_fn, mse = result
        y_pred = pred_fn(X)
        if np.all(np.isfinite(y_pred)):
            score = _nse_score(y_pred, y, var_y)
            candidates.append((pred_fn, score))
    
    # 6. Gaussian fit
    result = _gauss_fit(X, y)
    if result is not None:
        pred_fn, mse = result
        y_pred = pred_fn(X)
        if np.all(np.isfinite(y_pred)):
            score = _nse_score(y_pred, y, var_y)
            candidates.append((pred_fn, score))
    
    # 7. Higher-degree polynomial with more terms (for 1D)
    for deg in [10, 12, 15]:
        try:
            w, degree = _poly_fit(X, y, deg)
            y_pred = _poly_predict_from_w(X, w, deg)
            if np.all(np.isfinite(y_pred)):
                score = _nse_score(y_pred, y, var_y)
                candidates.append((lambda Xt, w=w, d=deg: _poly_predict_from_w(Xt, w, d), score))
        except:
            pass
    
    # If no candidates, use simple linear
    if not candidates:
        w, degree = _poly_fit(X, y, 1)
        def predict(Xt):
            return _poly_predict_from_w(Xt, w, degree)
        return predict
    
    # Sort by score and pick best
    candidates.sort(key=lambda x: x[1])
    best_pred, best_score = candidates[0]
    
    # If best score is very good, return it
    if best_score < 1e-6:
        def predict(Xt):
            Xt = np.asarray(Xt, dtype=float)
            result = best_pred(Xt)
            if not np.all(np.isfinite(result)):
                # fallback to polynomial
                w, degree = _poly_fit(X, y, 3)
                result = _poly_predict_from_w(Xt, w, degree)
            return result
        return predict
    
    # If best is not great, try ensemble of top candidates
    if len(candidates) >= 2 and candidates[1][1] < best_score * 1.5:
        # Average top 2-3 candidates
        top_n = min(3, len(candidates))
        top_preds = [c[0] for c in candidates[:top_n]]
        top_scores = [c[1] for c in candidates[:top_n]]
        # Weighted average by inverse score
        weights = [1.0 / max(s, 1e-10) for s in top_scores]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
        
        def predict(Xt):
            Xt = np.asarray(Xt, dtype=float)
            result = np.zeros(len(Xt))
            for wf, pf in zip(weights, top_preds):
                try:
                    r = pf(Xt)
                    if np.all(np.isfinite(r)):
                        result += wf * r
                    else:
                        # fallback
                        w_poly, deg_poly = _poly_fit(X, y, 3)
                        result += wf * _poly_predict_from_w(Xt, w_poly, deg_poly)
                except:
                    w_poly, deg_poly = _poly_fit(X, y, 3)
                    result += wf * _poly_predict_from_w(Xt, w_poly, deg_poly)
            return result
        return predict
    
    # Return best single candidate
    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        result = best_pred(Xt)
        if not np.all(np.isfinite(result)):
            w_poly, deg_poly = _poly_fit(X, y, 3)
            result = _poly_predict_from_w(Xt, w_poly, deg_poly)
        return result
    return predict
# EVOLVE-BLOCK-END
