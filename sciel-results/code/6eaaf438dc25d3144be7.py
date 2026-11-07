import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize
from scipy.interpolate import CubicSpline
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
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

    # 1. Try Polynomials (Degrees 1 to 15)
    for deg in range(1, 16):
        if time.time() > deadline - 5:
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

    # 2. Try Rational Functions: P(x) / Q(x)
    # P(x) = sum_{i=0}^{np} a_i x^i
    # Q(x) = 1 + sum_{i=1}^{nq} b_i x^i (normalize constant term to 1)
    def rational_func(x, *params):
        np_deg = int(len(params) / 2)
        nq_deg = int(len(params) / 2)
        a = params[:np_deg]
        b = params[np_deg:]
        
        # Build numerator
        num = np.zeros_like(x)
        for i, ai in enumerate(a):
            num += ai * (x ** i)
            
        # Build denominator
        den = np.ones_like(x)
        for i, bi in enumerate(b):
            den += bi * (x ** (i + 1))
            
        # Avoid division by zero
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        
        return num / den

    for np_deg in [1, 2, 3, 4]:
        for nq_deg in [1, 2, 3, 4]:
            if time.time() > deadline - 5:
                break
            try:
                # Initial guess
                p0 = np.zeros(np_deg + nq_deg)
                # Scale initial guess based on data
                p0[:np_deg] = np.linspace(0.1, 1.0, np_deg)
                p0[np_deg:] = np.linspace(0.1, 0.5, nq_deg)
                
                # Try multiple initializations if first fails or is bad
                for init_attempt in range(3):
                    if init_attempt > 0:
                        p0 = np.random.normal(0, 0.1, size=(np_deg + nq_deg))
                        p0[:np_deg] += np.linspace(0.1, 1.0, np_deg)
                        
                    popt, pcov = curve_fit(rational_func, X, y, p0=p0, maxfev=5000)
                    y_pred = rational_func(X, *popt)
                    if not np.all(np.isfinite(y_pred)):
                        continue
                    mse = np.mean((y_pred - y)**2)
                    
                    if mse < best_mse:
                        best_mse = mse
                        p = popt.copy()
                        def predict_rat(X, p=p):
                            x = np.asarray(X).flatten()
                            return rational_func(x, *p).reshape(np.asarray(X).shape)
                        best_model = predict_rat
                        
                    if mse < 1e-10:
                        return best_model
            except Exception:
                continue

    # 3. Try Small Neural Network (Universal Approximator)
    # Architecture: Input -> Hidden1 (tanh) -> Hidden2 (tanh) -> Output (linear)
    # Parameters: W1, b1, W2, b2, W3, b3
    
    def nn_func(x, *params):
        # Flatten x for NN
        x_flat = np.asarray(x).flatten()
        
        # Define sizes
        h1 = 8
        h2 = 8
        
        # Split params
        w1 = params[0:h1].reshape(1, h1) # 1 input
        b1 = params[h1:2*h1]
        w2 = params[2*h1:2*h1+h1*h2].reshape(h1, h2)
        b2 = params[2*h1+h1*h2:2*h1+h1*h2+h2]
        w3 = params[2*h1+h1*h2+h2:2*h1+h1*h2+2*h2].reshape(h2, 1)
        b3 = params[2*h1+h1*h2+2*h2:2*h1+h1*h2+2*h2+1]
        
        # Forward pass
        z1 = x_flat.reshape(-1, 1) @ w1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ w2 + b2
        a2 = np.tanh(z2)
        z3 = a2 @ w3 + b3
        
        return z3.flatten()

    try:
        if time.time() < deadline - 30:
            h1, h2 = 8, 8
            n_params = 1*h1 + h1 + h1*h2 + h2 + h2*1 + 1
            # Initialize with small random weights
            np.random.seed(42)
            p0 = np.random.randn(n_params) * 0.1
            
            # Use least_squares for stability
            def loss(params):
                try:
                    pred = nn_func(X, *params)
                    if not np.all(np.isfinite(pred)):
                        return 1e10
                    return pred - y
                except:
                    return 1e10 * np.ones_like(y)

            res = least_squares(loss, p0, max_nfev=1000)
            y_pred = nn_func(X, *res.x)
            if np.all(np.isfinite(y_pred)):
                mse = np.mean((y_pred - y)**2)
                if mse < best_mse:
                    best_mse = mse
                    p = res.x.copy()
                    def predict_nn(X, p=p):
                        x = np.asarray(X).flatten()
                        return nn_func(x, *p).reshape(np.asarray(X).shape)
                    best_model = predict_nn
                    
    except Exception:
        pass

    # 4. Cubic Spline Interpolation (Fallback for smooth unknowns)
    try:
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        # Handle duplicates
        unique_X, inverse_indices = np.unique(X_sorted, return_inverse=True)
        if len(unique_X) < n:
            unique_y = np.zeros(len(unique_X))
            counts = np.zeros(len(unique_X))
            np.add.at(unique_y, inverse_indices, y_sorted)
            np.add.at(counts, inverse_indices, 1)
            unique_y /= counts
            X_sorted = unique_X
            y_sorted = unique_y
            
        cs = CubicSpline(X_sorted, y_sorted, bc_type='natural')
        y_pred = cs(X)
        mse = np.mean((y_pred - y)**2)
        
        if mse < best_mse:
            best_mse = mse
            cs_copy = cs
            def predict_spline(X):
                x = np.asarray(X).flatten()
                return cs_copy(x).reshape(np.asarray(X).shape)
            best_model = predict_spline
            
    except Exception:
        pass

    # 5. Fallback: Mean
    if best_model is None:
        mean_y = np.mean(y)
        def predict_mean(X):
            return np.full(np.asarray(X).shape, mean_y)
        best_model = predict_mean

    return best_model
