import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import CubicSpline
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    X = np.asarray(X_train).flatten().astype(float)
    y = np.asarray(y_train).flatten().astype(float)
    n = len(X)
    
    # Handle constant case
    if np.var(y) < 1e-12:
        mean_y = np.mean(y)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const

    best_model = None
    best_mse = np.inf
    
    def update_best(mse, predict_func):
        nonlocal best_model, best_mse
        if mse < best_mse:
            best_mse = mse
            best_model = predict_func

    # 1. Try Polynomials (Degrees 1 to 12)
    for deg in range(1, 13):
        if time.time() - start_time > deadline - 10:
            break
        try:
            if deg > n - 1:
                break
            coef = np.polyfit(X, y, deg)
            y_pred = np.polyval(coef, X)
            mse = np.mean((y_pred - y)**2)
            
            if mse < best_mse:
                best_mse = mse
                c = coef.copy()
                def predict_poly(X, c=c):
                    x = np.asarray(X).flatten()
                    return np.polyval(c, x).reshape(np.asarray(X).shape)
                best_model = predict_poly
                
            if mse < 1e-10:
                return best_model
        except Exception:
            continue

    # 2. Try Parametric Functions
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    x_range = x_max - x_min
    
    if x_range < 1e-6:
        mean_y = np.mean(y)
        def predict_const_x(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const_x

    candidates = []
    
    # Sine: A*sin(B*x + C) + D
    def func_sin(x, A, B, C, D):
        return A * np.sin(B * x + C) + D
    candidates.append((func_sin, [y_range, 2*np.pi/x_range, 0, (y_min+y_max)/2]))
    
    # Cosine: A*cos(B*x + C) + D
    def func_cos(x, A, B, C, D):
        return A * np.cos(B * x + C) + D
    candidates.append((func_cos, [y_range, 2*np.pi/x_range, 0, (y_min+y_max)/2]))
    
    # Exponential: A*exp(B*x) + C
    def func_exp(x, A, B, C):
        return A * np.exp(B * x) + C
    candidates.append((func_exp, [y_range, 1.0/x_range, y_min]))
    
    # Power Law: A*x^B + C
    if x_min > 0.01:
        def func_pow(x, A, B, C):
            return A * np.power(x, B) + C
        candidates.append((func_pow, [1.0, 1.0, y_min]))
        
        # Log: A*log(x) + B
        def func_log(x, A, B):
            return A * np.log(x) + B
        candidates.append((func_log, [y_range, y_min]))
        
        # Rational: A/(x+B) + C
        def func_rat(x, A, B, C):
            return A / (x + B) + C
        candidates.append((func_rat, [y_range*x_range, -x_min*0.1, y_min]))
        
    # Gaussian: A*exp(-B*x^2) + C
    def func_gauss(x, A, B, C):
        return A * np.exp(-B * x**2) + C
    candidates.append((func_gauss, [y_range, 1.0/(x_range**2), y_min]))
    
    # Hyperbolic Tangent: A*tanh(B*x) + C
    def func_tanh(x, A, B, C):
        return A * np.tanh(B * x) + C
    candidates.append((func_tanh, [y_range, 1.0/x_range, (y_min+y_max)/2]))
    
    # Absolute value: A*|x| + B
    def func_abs(x, A, B):
        return A * np.abs(x) + B
    candidates.append((func_abs, [y_range/x_range, y_min]))
    
    for func, p0 in candidates:
        if time.time() - start_time > deadline - 10:
            break
        try:
            popt, pcov = curve_fit(func, X, y, p0=p0, maxfev=2000)
            y_pred = func(X, *popt)
            mse = np.mean((y_pred - y)**2)
            
            if mse < best_mse:
                best_mse = mse
                f = func
                p = popt.copy()
                def predict_param(X, f=f, p=p):
                    x = np.asarray(X).flatten()
                    return f(x, *p).reshape(np.asarray(X).shape)
                best_model = predict_param
                
            if mse < 1e-10:
                return best_model
        except Exception:
            continue

    # 3. Try a combination of basis functions via Linear Least Squares
    if time.time() - start_time < deadline - 20:
        try:
            cols = [np.ones_like(X)]
            cols.append(X)
            cols.append(X**2)
            cols.append(X**3)
            cols.append(X**4)
            cols.append(np.sin(X))
            cols.append(np.cos(X))
            cols.append(np.sin(2*X))
            cols.append(np.cos(2*X))
            cols.append(np.exp(X))
            cols.append(np.exp(-X))
            if np.all(X > 0.01):
                cols.append(1.0/X)
                cols.append(np.log(X))
            
            A_matrix = np.column_stack(cols)
            coeffs, residuals, rank, s = np.linalg.lstsq(A_matrix, y, rcond=None)
            y_pred = A_matrix @ coeffs
            mse = np.mean((y_pred - y)**2)
            
            if mse < best_mse:
                best_mse = mse
                A_m = A_matrix
                c = coeffs.copy()
                
                def predict_basis(X):
                    x = np.asarray(X).flatten()
                    cols_p = [np.ones_like(x)]
                    cols_p.append(x)
                    cols_p.append(x**2)
                    cols_p.append(x**3)
                    cols_p.append(x**4)
                    cols_p.append(np.sin(x))
                    cols_p.append(np.cos(x))
                    cols_p.append(np.sin(2*x))
                    cols_p.append(np.cos(2*x))
                    cols_p.append(np.exp(x))
                    cols_p.append(np.exp(-x))
                    if np.all(x > 0.01):
                        cols_p.append(1.0/x)
                        cols_p.append(np.log(x))
                    A_p = np.column_stack(cols_p)
                    return (A_p @ c).reshape(np.asarray(X).shape)
                
                best_model = predict_basis
        except Exception:
            pass

    # 4. Cubic Spline Interpolation (Key Improvement from Inspiration)
    # Splines can fit arbitrary smooth functions well, especially if the function is not captured by the previous bases.
    # We sort the data first as spline requires increasing x.
    try:
        # Sort by X
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        # Check for duplicate X values. If present, spline might fail or be ill-conditioned.
        # For 200 points in a continuous range, duplicates are unlikely unless the function is multi-valued (not a function) or discrete.
        # The task says "regression algorithm", implying y = f(x).
        # If duplicates exist, we can average y for same x to be safe, but let's try direct spline first.
        # CubicSpline requires unique x values.
        if len(np.unique(X_sorted)) < n:
             # Handle duplicates by averaging
             unique_X, inverse_indices = np.unique(X_sorted, return_inverse=True)
             unique_y = np.zeros(len(unique_X))
             counts = np.zeros(len(unique_X))
             np.add.at(unique_y, inverse_indices, y_sorted)
             np.add.at(counts, inverse_indices, 1)
             unique_y /= counts
             X_sorted = unique_X
             y_sorted = unique_y
             
        # Fit Cubic Spline
        # bc_type='natural' is a common choice, 'not-a-knot' is another.
        # Let's try 'natural' first as it's smooth.
        cs = CubicSpline(X_sorted, y_sorted, bc_type='natural')
        
        # Evaluate on training data to check MSE
        y_pred = cs(X)
        mse = np.mean((y_pred - y)**2)
        
        if mse < best_mse:
            best_mse = mse
            cs_copy = cs
            def predict_spline(X):
                x = np.asarray(X).flatten()
                return cs_copy(x).reshape(np.asarray(X).shape)
            best_model = predict_spline
            
            if mse < 1e-10:
                return best_model
                
    except Exception:
        # Spline failed, maybe due to duplicates or ill-conditioning
        pass

    # 5. Fallback: High degree polynomial interpolation
    if best_model is None or best_mse > 1e-4 * np.var(y):
        try:
            deg = min(10, n-1)
            coef = np.polyfit(X, y, deg)
            y_pred = np.polyval(coef, X)
            mse = np.mean((y_pred - y)**2)
            
            c = coef.copy()
            def predict_fallback(X, c=c):
                x = np.asarray(X).flatten()
                return np.polyval(c, x).reshape(np.asarray(X).shape)
            
            if mse < best_mse:
                best_model = predict_fallback
                best_mse = mse
        except Exception:
            mean_y = np.mean(y)
            def predict_mean(X):
                return np.full(np.asarray(X).shape, mean_y)
            best_model = predict_mean

    return best_model
