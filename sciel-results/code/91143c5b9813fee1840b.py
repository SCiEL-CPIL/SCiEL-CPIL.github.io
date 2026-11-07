"""Rewritten program for symbolic regression with multi-strategy ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def fit(X_train, y_train):
    start_time = time.time()
    time_limit = 250.0  # Safety margin for 330s limit, 4 equations
    
    X = X_train
    y = y_train
    n_samples, n_features = X.shape
    
    # Ensure 1D for simplicity if needed, but task says 1 input
    if n_features > 1:
        # If multi-dimensional, we might need different handling, but task says 1 input.
        # We'll proceed assuming 1D or flatten.
        X_flat = X.flatten()
    else:
        X_flat = X[:, 0]
        
    # Store best model
    best_mse = np.inf
    best_predict_fn = None
    best_model_name = "None"
    
    # Helper to check time
    def time_check():
        return (time.time() - start_time) < time_limit

    # 1. Polynomial Regression
    # Degrees 2 to 12
    for deg in range(2, 13):
        if not time_check():
            break
        # Create Vandermonde-like matrix
        # Using np.vander for stability
        V = np.vander(X_flat, N=deg + 1, increasing=True)
        try:
            # Use Ridge to handle ill-conditioning for high degrees
            # Lambda small to allow flexibility but prevent blowup
            alpha = 1e-6
            A = V.T @ V + alpha * np.eye(deg + 1)
            b = V.T @ y
            w = np.linalg.solve(A, b)
            y_pred = V @ w
            mse = np.mean((y_pred - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_model_name = f"Poly(deg={deg})"
                # Capture w for prediction
                w_copy = w.copy()
                deg_copy = deg
                def make_poly_pred(w, deg):
                    def predict(X_test):
                        Xf = X_test[:, 0] if X_test.ndim > 1 else X_test
                        Vt = np.vander(Xf, N=deg + 1, increasing=True)
                        return Vt @ w
                    return predict
                best_predict_fn = make_poly_pred(w_copy, deg_copy)
        except np.linalg.LinAlgError:
            continue

    # 2. RBF Regression (Gaussian Kernel)
    # Try different bandwidths (sigma)
    # gamma = 1 / (2 * sigma^2)
    # Let's try sigmas: 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
    sigmas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    for sigma in sigmas:
        if not time_check():
            break
        gamma = 1.0 / (2.0 * sigma**2)
        # Compute RBF matrix K
        # K[i,j] = exp(-gamma * (x_i - x_j)^2)
        # Efficient computation
        sq_dists = (X_flat[:, None] - X_flat[None, :])**2
        K = np.exp(-gamma * sq_dists)
        
        # Ridge regression: (K + lambda*I) w = y
        # Choose lambda based on sigma? Smaller sigma (narrower) -> higher variance -> larger lambda?
        # Let's try a few lambdas for each sigma
        lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
        for lam in lambdas:
            if not time_check():
                break
            A = K + lam * np.eye(n_samples)
            try:
                w = np.linalg.solve(A, y)
                y_pred = K @ w
                mse = np.mean((y_pred - y) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_model_name = f"RBF(sigma={sigma}, lam={lam})"
                    w_copy = w.copy()
                    gamma_copy = gamma
                    lam_copy = lam
                    def make_rbf_pred(w, gamma, lam):
                        def predict(X_test):
                            Xf = X_test[:, 0] if X_test.ndim > 1 else X_test
                            # Compute K_test
                            sq_dists_test = (Xf[:, None] - X_flat[None, :])**2
                            K_test = np.exp(-gamma * sq_dists_test)
                            return K_test @ w
                        return predict
                    best_predict_fn = make_rbf_pred(w_copy, gamma_copy, lam_copy)
            except np.linalg.LinAlgError:
                continue

    # 3. Fourier Series
    # Assume range is roughly [-3, 3] or similar.
    # Period T. Let's estimate T from data range.
    x_min, x_max = np.min(X_flat), np.max(X_flat)
    range_width = x_max - x_min
    if range_width < 1e-6:
        range_width = 1.0
    
    # Try max frequencies 1 to 20
    for k_max in range(1, 21):
        if not time_check():
            break
        # Basis functions: sin(2*pi*k*x/T), cos(2*pi*k*x/T)
        # Plus constant
        # T = range_width? Or maybe 2*range_width?
        # Let's try T = range_width and T = 2*range_width
        for T_mult in [1.0, 2.0]:
            T = range_width * T_mult
            if T < 1e-6:
                continue
            # Build feature matrix
            # Columns: 1, cos(2pi*1*x/T), sin(2pi*1*x/T), ..., cos(2pi*k_max*x/T), sin(2pi*k_max*x/T)
            cols = [np.ones(n_samples)]
            for k in range(1, k_max + 1):
                arg = 2 * np.pi * k * X_flat / T
                cols.append(np.cos(arg))
                cols.append(np.sin(arg))
            V = np.column_stack(cols)
            
            # Ridge regression
            alpha = 1e-6
            A = V.T @ V + alpha * np.eye(V.shape[1])
            b = V.T @ y
            try:
                w = np.linalg.solve(A, b)
                y_pred = V @ w
                mse = np.mean((y_pred - y) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_model_name = f"Fourier(k_max={k_max}, T_mult={T_mult})"
                    w_copy = w.copy()
                    k_max_copy = k_max
                    T_copy = T
                    def make_fourier_pred(w, k_max, T):
                        def predict(X_test):
                            Xf = X_test[:, 0] if X_test.ndim > 1 else X_test
                            cols = [np.ones(len(Xf))]
                            for k in range(1, k_max + 1):
                                arg = 2 * np.pi * k * Xf / T
                                cols.append(np.cos(arg))
                                cols.append(np.sin(arg))
                            Vt = np.column_stack(cols)
                            return Vt @ w
                        return predict
                    best_predict_fn = make_fourier_pred(w_copy, k_max_copy, T_copy)
            except np.linalg.LinAlgError:
                continue

    # 4. Mixed Basis (Heuristic)
    # Features: 1, x, x^2, x^3, x^4, sin(x), cos(x), sin(2x), cos(2x), exp(x), exp(-x), tanh(x)
    if time_check():
        def build_mixed_features(Xf):
            cols = [
                np.ones(len(Xf)),
                Xf,
                Xf**2,
                Xf**3,
                Xf**4,
                np.sin(Xf),
                np.cos(Xf),
                np.sin(2*Xf),
                np.cos(2*Xf),
                np.exp(Xf),
                np.exp(-Xf),
                np.tanh(Xf)
            ]
            return np.column_stack(cols)
        
        V_train = build_mixed_features(X_flat)
        # Ridge
        alpha = 1e-6
        A = V_train.T @ V_train + alpha * np.eye(V_train.shape[1])
        b = V_train.T @ y
        try:
            w = np.linalg.solve(A, b)
            y_pred = V_train @ w
            mse = np.mean((y_pred - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_model_name = "MixedBasis"
                w_copy = w.copy()
                def make_mixed_pred(w):
                    def predict(X_test):
                        Xf = X_test[:, 0] if X_test.ndim > 1 else X_test
                        Vt = build_mixed_features(Xf)
                        return Vt @ w
                    return predict
                best_predict_fn = make_mixed_pred(w_copy)
        except np.linalg.LinAlgError:
            pass

    # If no model was found (should not happen), fallback to mean
    if best_predict_fn is None:
        mean_y = np.mean(y)
        def predict_mean(X_test):
            return np.full(X_test.shape[0], mean_y)
        return predict_mean

    return best_predict_fn

# EVOLVE-BLOCK-END
