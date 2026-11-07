import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import PchipInterpolator, CubicSpline
from sklearn.linear_model import Ridge
import time
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin
    
    X = np.asarray(X_train).flatten().astype(float)
    y = np.asarray(y_train).flatten().astype(float)
    n = len(X)
    
    # Handle constant case
    if np.var(y) < 1e-12:
        mean_y = np.mean(y)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const

    # 1. Prepare Data
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) < n:
        unique_y = np.zeros(len(unique_X))
        np.add.at(unique_y, inverse_indices, y_sorted)
        unique_y /= counts
        X_sorted = unique_X
        y_sorted = unique_y
        
    X_fit = X_sorted
    y_fit = y_sorted
    n_fit = len(X_fit)
    
    best_model = None
    best_mse = np.inf

    def update_best(mse, predict_func):
        nonlocal best_model, best_mse
        if mse < best_mse:
            best_mse = mse
            best_model = predict_func

    # 2. Interpolation Baselines
    # PCHIP
    try:
        pchip = PchipInterpolator(X_fit, y_fit)
        y_pred = pchip(X)
        if np.all(np.isfinite(y_pred)):
            mse = np.mean((y_pred - y)**2)
            if mse < best_mse:
                best_mse = mse
                pchip_copy = pchip
                def predict_pchip(X):
                    x = np.asarray(X).flatten()
                    return pchip_copy(x).reshape(np.asarray(X).shape)
                best_model = predict_pchip
    except Exception:
        pass

    # Cubic Spline
    try:
        cs = CubicSpline(X_fit, y_fit, bc_type='natural')
        y_pred = cs(X)
        if np.all(np.isfinite(y_pred)):
            mse = np.mean((y_pred - y)**2)
            if mse < best_mse:
                best_mse = mse
                cs_copy = cs
                def predict_cs(X):
                    x = np.asarray(X).flatten()
                    return cs_copy(x).reshape(np.asarray(X).shape)
                best_model = predict_cs
    except Exception:
        pass

    # 3. Polynomial Fits (Low to Mid degree)
    for deg in range(1, 16):
        if time.time() > deadline - 10:
            break
        try:
            if deg >= n_fit:
                break
            coef = np.polyfit(X_fit, y_fit, deg)
            y_pred = np.polyval(coef, X)
            if not np.all(np.isfinite(y_pred)):
                continue
            mse = np.mean((y_pred - y)**2)
            if mse < best_mse:
                best_mse = mse
                c = coef.copy()
                def predict_poly(X, c=c):
                    x = np.asarray(X).flatten()
                    return np.polyval(c, x).reshape(np.asarray(X).shape)
                best_model = predict_poly
            if mse < 1e-12:
                return best_model
        except Exception:
            continue

    # 4. Linear Basis Search (Regularized Least Squares)
    # This attempts to find a linear combination of common functions.
    # Basis: 1, x, x^2, ..., x^10, sin(x), cos(x), sin(2x), cos(2x), exp(x), log(|x|+1), 1/sqrt(|x|+1)
    try:
        if time.time() < deadline - 30:
            basis_funcs = []
            basis_names = []
            
            # Polynomials
            for d in range(0, 11):
                basis_funcs.append(lambda x, d=d: x**d)
                basis_names.append(f"x^{d}")
            
            # Trig
            for freq in [1, 2, 3]:
                basis_funcs.append(lambda x, f=freq: np.sin(f * x))
                basis_names.append(f"sin({f}x)")
                basis_funcs.append(lambda x, f=freq: np.cos(f * x))
                basis_names.append(f"cos({f}x)")
            
            # Exp
            basis_funcs.append(lambda x: np.exp(x))
            basis_names.append("exp(x)")
            basis_funcs.append(lambda x: np.exp(-x))
            basis_names.append("exp(-x)")
            
            # Log
            basis_funcs.append(lambda x: np.log(np.abs(x) + 1e-6))
            basis_names.append("log(|x|)")
            
            # Power
            basis_funcs.append(lambda x: np.sqrt(np.abs(x) + 1e-6))
            basis_names.append("sqrt(|x|)")
            basis_funcs.append(lambda x: 1.0 / (np.abs(x) + 1e-6))
            basis_names.append("1/|x|")

            # Build Design Matrix
            # We need to be careful with numerical stability. 
            # Normalize columns? Ridge regression handles scaling somewhat, but explicit scaling is better.
            cols = []
            for f in basis_funcs:
                try:
                    col = f(X_fit)
                    if not np.all(np.isfinite(col)):
                        col = np.where(np.isfinite(col), col, 0)
                    cols.append(col)
                except Exception:
                    cols.append(np.zeros_like(X_fit))
            
            A = np.column_stack(cols)
            
            # Standardize A for stability
            mean_A = np.mean(A, axis=0)
            std_A = np.std(A, axis=0)
            std_A[std_A == 0] = 1.0
            A_norm = (A - mean_A) / std_A
            
            # Fit Ridge
            # Use a range of alphas to find the best trade-off
            best_ridge_mse = np.inf
            best_ridge_coef = None
            best_alpha = None
            
            # Note: Ridge fits the normalized data. We need to transform back.
            # y_pred_norm = w @ A_norm
            # y_pred = w @ (A - mean_A)/std_A
            # We fit y to A_norm.
            
            for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
                try:
                    ridge = Ridge(alpha=alpha)
                    ridge.fit(A_norm, y_fit)
                    w = ridge.coef_
                    b = ridge.intercept_
                    
                    # Predict on train
                    y_pred = w @ A_norm.T + b
                    if not np.all(np.isfinite(y_pred)):
                        continue
                    mse = np.mean((y_pred - y_fit)**2)
                    if mse < best_ridge_mse:
                        best_ridge_mse = mse
                        best_ridge_coef = (w, b)
                        best_alpha = alpha
                except Exception:
                    continue
            
            if best_ridge_coef is not None and best_ridge_mse < best_mse:
                w, b = best_ridge_coef
                def predict_ridge(X):
                    x = np.asarray(X).flatten()
                    # Reconstruct A for new X
                    new_cols = []
                    for f in basis_funcs:
                        try:
                            col = f(x)
                            if not np.all(np.isfinite(col)):
                                col = np.where(np.isfinite(col), col, 0)
                            new_cols.append(col)
                        except Exception:
                            new_cols.append(np.zeros_like(x))
                    A_new = np.column_stack(new_cols)
                    A_new_norm = (A_new - mean_A) / std_A
                    return (w @ A_new_norm.T + b).reshape(np.asarray(X).shape)
                
                # Verify on full train set X (not just sorted unique)
                y_pred_full = predict_ridge(X)
                mse_full = np.mean((y_pred_full - y)**2)
                if np.all(np.isfinite(y_pred_full)) and mse_full < best_mse:
                    best_mse = mse_full
                    best_model = predict_ridge

    except Exception:
        pass

    # 5. Non-linear Parametric Fits with Multi-start
    # Forms: A*sin(B*x+C)+D, A*cos(B*x+C)+D, A*exp(B*x)+C, A*x**B+C, A*log(B*x+C)+D
    
    def make_predict(func, params):
        def predict(X):
            x = np.asarray(X).flatten()
            return func(x, *params).reshape(np.asarray(X).shape)
        return predict

    def try_fit(func, p0_list, bounds=None, name=""):
        nonlocal best_mse, best_model
        if time.time() > deadline - 5:
            return
        for p0 in p0_list:
            try:
                if bounds:
                    popt, _ = curve_fit(func, X_fit, y_fit, p0=p0, bounds=bounds, maxfev=2000)
                else:
                    popt, _ = curve_fit(func, X_fit, y_fit, p0=p0, maxfev=2000)
                
                y_pred = func(X, *popt)
                if not np.all(np.isfinite(y_pred)):
                    continue
                mse = np.mean((y_pred - y)**2)
                if mse < best_mse:
                    best_mse = mse
                    p = popt.copy()
                    best_model = make_predict(func, p)
                if mse < 1e-12:
                    return
            except Exception:
                continue

    # Define functions
    def sin_func(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    def cos_func(x, A, B, C, D):
        return A * np.cos(B * x + C) + D

    def exp_func(x, A, B, C):
        return A * np.exp(B * x) + C

    def power_func(x, A, B, C):
        return A * np.power(np.abs(x) + 1e-6, B) + C

    def log_func(x, A, B, C, D):
        return A * np.log(B * x + C) + D

    # Initial guesses
    y_mean = np.mean(y_fit)
    y_std = np.std(y_fit)
    x_range = X_fit.max() - X_fit.min()
    
    # Sin/Cos
    p0_sin = [
        [y_std, 1.0, 0.0, y_mean],
        [y_std, 2.0, 0.0, y_mean],
        [y_std, 1.0, np.pi/2, y_mean],
        [y_std * 0.5, 1.0, 0.0, y_mean],
        [y_std * 2.0, 0.5, 0.0, y_mean],
    ]
    try_fit(sin_func, p0_sin, bounds=([-1000, 0.01, -10, -10000], [1000, 10, 10, 10000]))
    
    p0_cos = [
        [y_std, 1.0, 0.0, y_mean],
        [y_std, 2.0, 0.0, y_mean],
        [y_std, 1.0, np.pi, y_mean],
    ]
    try_fit(cos_func, p0_cos, bounds=([-1000, 0.01, -10, -10000], [1000, 10, 10, 10000]))

    # Exp
    # A * exp(B*x) + C
    # If x is in [-3,3], exp(x) is in [0.05, 20].
    p0_exp = [
        [y_std, 1.0, y_mean - y_std * np.exp(1)],
        [y_std, 0.5, y_mean - y_std * np.exp(0.5)],
        [-y_std, 1.0, y_mean + y_std * np.exp(1)],
        [y_std / 10, 2.0, y_mean],
    ]
    try_fit(exp_func, p0_exp, bounds=([-1000, -5, -10000], [1000, 5, 10000]))

    # Power
    p0_pow = [
        [y_std, 1.0, y_mean],
        [y_std, 2.0, y_mean],
        [y_std, 0.5, y_mean],
        [-y_std, 1.0, y_mean],
    ]
    try_fit(power_func, p0_pow, bounds=([-1000, 0.1, -10000], [1000, 5, 10000]))

    # Log
    # Only valid if B*x+C > 0. 
    # Assuming x in [-2,2] or similar.
    p0_log = [
        [y_std, 1.0, 3.0, y_mean],
        [y_std, 0.5, 5.0, y_mean],
        [-y_std, 1.0, 3.0, y_mean],
    ]
    try_fit(log_func, p0_log, bounds=([-1000, 0.01, 0.1, -10000], [1000, 5, 100, 10000]))

    # 6. Fallback
    if best_model is None:
        mean_y = np.mean(y)
        def predict_mean(X):
            return np.full(np.asarray(X).shape, mean_y)
        best_model = predict_mean

    return best_model
