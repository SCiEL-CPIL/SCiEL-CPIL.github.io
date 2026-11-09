import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, cholesky

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 295  # Safety margin before 330s kill
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    # Flatten for 1D
    if X.ndim == 2:
        x = X[:, 0]
    else:
        x = X.flatten()
    y = y.flatten()
    
    n = len(x)
    
    # Normalize x to [0, 1] for better GP conditioning
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    x_norm = (x - x_min) / x_range
    
    # Sort by x for better Cholesky stability
    sort_idx = np.argsort(x_norm)
    x_sorted = x_norm[sort_idx]
    y_sorted = y[sort_idx]
    
    best_predict = None
    best_score = np.inf
    
    # Strategy 1: Gaussian Process with L-BFGS-B hyperparameter optimization
    # GP kernel: k(x, x') = sigma^2 * exp(-0.5 * (x - x')^2 / length_scale^2)
    
    def _gp_predict_factory(x_train_norm, y_train, length_scale, sigma, x_min, x_range, sort_idx):
        """Create a predict function for a GP with given hyperparameters."""
        n_train = len(x_train_norm)
        K = sigma**2 * np.exp(-0.5 * (x_train_norm[:, None] - x_train_norm[None, :])**2 / length_scale**2)
        # Add small jitter for numerical stability
        K += 1e-10 * np.eye(n_train)
        
        try:
            L = cholesky(K, lower=True)
            # Pre-compute alpha = K^{-1} y
            # K alpha = y  =>  L L^T alpha = y  =>  alpha = L^{-T} L^{-1} y
            alpha = cho_solve((L, True), y_train)
            
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=float)
                if X_test.ndim == 1:
                    xt = X_test
                else:
                    xt = X_test[:, 0]
                xt_norm = (xt - x_min) / x_range
                
                # k_test(x, x_train) for all test points
                diff = xt_norm[:, None] - x_train_norm[None, :]
                K_test = sigma**2 * np.exp(-0.5 * diff**2 / length_scale**2)
                
                return K_test @ alpha
            return predict
        except Exception:
            return None
    
    def _neg_log_marginal_likelihood(params):
        """Negative log marginal likelihood for GP."""
        log_sigma, log_ls = params
        sigma = np.exp(log_sigma)
        length_scale = np.exp(log_ls)
        
        K = sigma**2 * np.exp(-0.5 * (x_sorted[:, None] - x_sorted[None, :])**2 / length_scale**2)
        K += 1e-10 * np.eye(n)
        
        try:
            L = cholesky(K, lower=True)
            alpha = cho_solve((L, True), y_sorted)
            # NLL = 0.5 * y^T K^{-1} y + 0.5 * log det K + n/2 * log(2*pi)
            nll = 0.5 * y_sorted @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            return nll
        except Exception:
            return 1e10
    
    # Initial parameter guesses
    best_params = None
    best_nll = np.inf
    
    # Try several initial values for log_sigma and log_ls
    initial_guesses = [
        (0.0, -1.0),  # sigma=1, ls=0.37
        (0.0, 0.0),   # sigma=1, ls=1.0
        (0.5, -0.5),  # sigma=1.65, ls=0.6
        (-0.5, 0.5),  # sigma=0.6, ls=1.65
        (1.0, -1.5),  # sigma=2.7, ls=0.22
        (0.0, -2.0),  # sigma=1, ls=0.14
    ]
    
    for log_s0, log_ls0 in initial_guesses:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(_neg_log_marginal_likelihood, 
                          x0=[log_s0, log_ls0], 
                          method='L-BFGS-B',
                          bounds=[(-5, 5), (-5, 5)],
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except Exception:
            continue
    
    if best_params is not None:
        sigma = np.exp(best_params[0])
        length_scale = np.exp(best_params[1])
        
        gp_pred = _gp_predict_factory(x_sorted, y_sorted, length_scale, sigma, x_min, x_range, sort_idx)
        if gp_pred is not None:
            y_pred = gp_pred(x_sorted)
            mse = np.mean((y_pred - y_sorted)**2)
            var_y = np.var(y)
            if var_y < 1e-10:
                var_y = 1.0
            score = mse / var_y
            if score < best_score:
                best_score = score
                best_predict = gp_pred
    
    # Strategy 2: Polynomial fits as fallback
    for deg in [4, 6, 8, 10, 14, 18]:
        if time.time() > deadline - 30:
            break
        try:
            x_norm_full = (x - x_min) / x_range
            A = np.vander(x_norm_full, N=deg+1, increasing=True)
            lam = 1e-6
            coeffs = np.linalg.solve(A.T @ A + lam * np.eye(deg+1), A.T @ y)
            
            def make_poly_pred(coeffs, x_min, x_range):
                def predict(X_test):
                    X_test = np.asarray(X_test, dtype=float)
                    if X_test.ndim == 1:
                        xt = X_test
                    else:
                        xt = X_test[:, 0]
                    xt_norm = (xt - x_min) / x_range
                    result = np.full_like(xt_norm, coeffs[-1], dtype=float)
                    for c in coeffs[-2::-1]:
                        result = result * xt_norm + c
                    return result
                return predict
            
            pred = make_poly_pred(coeffs, x_min, x_range)
            y_pred = pred(x)
            mse = np.mean((y_pred - y)**2)
            var_y = np.var(y)
            if var_y < 1e-10:
                var_y = 1.0
            score = mse / var_y
            if score < best_score:
                best_score = score
                best_predict = pred
        except Exception:
            continue
    
    # Strategy 3: Fourier series
    for n_terms in [5, 10, 15, 20]:
        if time.time() > deadline - 20:
            break
        try:
            x_norm_full = (x - x_min) / x_range
            cols = [np.ones_like(x_norm_full)]
            for k in range(1, n_terms + 1):
                angle = 2 * np.pi * k * x_norm_full
                cols.append(np.sin(angle))
                cols.append(np.cos(angle))
            A = np.column_stack(cols)
            lam = 1e-6
            coeffs = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
            
            def make_fourier_pred(coeffs, n_terms, x_min, x_range):
                def predict(X_test):
                    X_test = np.asarray(X_test, dtype=float)
                    if X_test.ndim == 1:
                        xt = X_test
                    else:
                        xt = X_test[:, 0]
                    xt_norm = (xt - x_min) / x_range
                    cols = [np.ones_like(xt_norm)]
                    for k in range(1, n_terms + 1):
                        angle = 2 * np.pi * k * xt_norm
                        cols.append(np.sin(angle))
                        cols.append(np.cos(angle))
                    A_test = np.column_stack(cols)
                    return A_test @ coeffs
                return predict
            
            pred = make_fourier_pred(coeffs, n_terms, x_min, x_range)
            y_pred = pred(x)
            mse = np.mean((y_pred - y)**2)
            var_y = np.var(y)
            if var_y < 1e-10:
                var_y = 1.0
            score = mse / var_y
            if score < best_score:
                best_score = score
                best_predict = pred
        except Exception:
            continue
    
    # Fallback: Mean
    if best_predict is None:
        mean_y = np.mean(y)
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            if X_test.ndim == 1:
                return np.full(X_test.shape, mean_y)
            else:
                return np.full(X_test.shape[0], mean_y)
        best_predict = predict
    
    return best_predict
