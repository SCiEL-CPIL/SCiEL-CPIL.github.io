import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import PchipInterpolator
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0
    
    X = np.asarray(X_train).flatten().astype(float)
    y = np.asarray(y_train).flatten().astype(float)
    n = len(X)
    
    # Handle constant case
    if np.var(y) < 1e-12:
        mean_y = np.mean(y)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const

    # 1. Prepare Data for Interpolation
    # Sort by X for interpolation
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates by averaging
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) < n:
        unique_y = np.zeros(len(unique_X))
        np.add.at(unique_y, inverse_indices, y_sorted)
        unique_y /= counts
        X_sorted = unique_X
        y_sorted = unique_y
    
    best_model = None
    best_mse = np.inf

    def update_best(mse, predict_func):
        nonlocal best_model, best_mse
        if mse < best_mse:
            best_mse = mse
            best_model = predict_func

    # 2. Try PCHIP (Monotone Cubic Interpolation)
    # This is robust for noiseless data and handles local features well.
    try:
        pchip = PchipInterpolator(X_sorted, y_sorted)
        y_pred = pchip(X)
        if np.all(np.isfinite(y_pred)):
            mse = np.mean((y_pred - y)**2)
            if mse < best_mse:
                best_mse = mse
                pchip_copy = pchip
                def predict_pchip(X):
                    x = np.asarray(X).flatten()
                    # Clip to training range to avoid extrapolation issues if test is out of bounds
                    # However, PCHIP extrapolates linearly by default. 
                    # For safety, we can just let it extrapolate or clip. 
                    # Given the task, test points are likely in range or close.
                    return pchip_copy(x).reshape(np.asarray(X).shape)
                best_model = predict_pchip
    except Exception:
        pass

    # 3. Try Simple Parametric Fits
    # We try a few common forms. If they fit better than PCHIP (unlikely for complex, but possible for simple), we use them.
    # We strictly limit the complexity to avoid instability.
    
    # 3a. Polynomials (Low Degree)
    for deg in range(1, 6):
        if time.time() > deadline - 10:
            break
        try:
            if deg >= len(X_sorted):
                break
            coef = np.polyfit(X_sorted, y_sorted, deg)
            y_pred = np.polyval(coef, X)
            if not np.all(np.isfinite(y_pred)):
                continue
            mse = np.mean((y_pred - y)**2)
            # Only update if it's better than current best (which includes PCHIP)
            if mse < best_mse:
                best_mse = mse
                c = coef.copy()
                def predict_poly(X, c=c):
                    x = np.asarray(X).flatten()
                    return np.polyval(c, x).reshape(np.asarray(X).shape)
                best_model = predict_poly
        except Exception:
            continue

    # 3b. Exponential / Logarithmic / Power
    # Forms: A * exp(B*x), A * log(B*x + C), A * x**B + C
    # We use curve_fit with careful bounds and initial guesses.
    
    def exp_func(x, A, B):
        return A * np.exp(B * x)
    
    def log_func(x, A, B, C):
        # Avoid log(0) or negative
        return A * np.log(np.abs(B * x + C) + 1e-12)
        
    def power_func(x, A, B, C):
        return A * np.power(np.abs(x) + 1e-12, B) + C

    # Try Exp
    try:
        if time.time() < deadline - 20:
            # Initial guess
            A0 = np.max(y_sorted) / np.max(np.exp(X_sorted)) if np.max(np.exp(X_sorted)) > 0 else 1
            B0 = 1.0
            popt, _ = curve_fit(exp_func, X_sorted, y_sorted, p0=[A0, B0], maxfev=2000)
            y_pred = exp_func(X, *popt)
            if np.all(np.isfinite(y_pred)):
                mse = np.mean((y_pred - y)**2)
                if mse < best_mse:
                    best_mse = mse
                    p = popt.copy()
                    def predict_exp(X, p=p):
                        x = np.asarray(X).flatten()
                        return exp_func(x, *p).reshape(np.asarray(X).shape)
                    best_model = predict_exp
    except Exception:
        pass

    # Try Log
    try:
        if time.time() < deadline - 20:
            # Initial guess
            A0 = 1.0
            B0 = 1.0
            C0 = 1.0
            popt, _ = curve_fit(log_func, X_sorted, y_sorted, p0=[A0, B0, C0], maxfev=2000)
            y_pred = log_func(X, *popt)
            if np.all(np.isfinite(y_pred)):
                mse = np.mean((y_pred - y)**2)
                if mse < best_mse:
                    best_mse = mse
                    p = popt.copy()
                    def predict_log(X, p=p):
                        x = np.asarray(X).flatten()
                        return log_func(x, *p).reshape(np.asarray(X).shape)
                    best_model = predict_log
    except Exception:
        pass

    # Try Power
    try:
        if time.time() < deadline - 20:
            A0 = 1.0
            B0 = 1.0
            C0 = 0.0
            popt, _ = curve_fit(power_func, X_sorted, y_sorted, p0=[A0, B0, C0], maxfev=2000)
            y_pred = power_func(X, *popt)
            if np.all(np.isfinite(y_pred)):
                mse = np.mean((y_pred - y)**2)
                if mse < best_mse:
                    best_mse = mse
                    p = popt.copy()
                    def predict_pow(X, p=p):
                        x = np.asarray(X).flatten()
                        return power_func(x, *p).reshape(np.asarray(X).shape)
                    best_model = predict_pow
    except Exception:
        pass

    # 4. Fallback: Mean
    if best_model is None:
        mean_y = np.mean(y)
        def predict_mean(X):
            return np.full(np.asarray(X).shape, mean_y)
        best_model = predict_mean

    return best_model
