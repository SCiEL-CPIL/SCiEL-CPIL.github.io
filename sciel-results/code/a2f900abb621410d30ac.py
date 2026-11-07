import numpy as np
import time
from scipy.linalg import cho_factor, cho_solve, solve

def _loocv_ridge(K, y, lam):
    """Compute LOOCV MSE for kernel ridge regression using the closed-form formula."""
    N = len(y)
    A = K + lam * np.eye(N)
    try:
        w = solve(A, y, assume_a='sym')
        # Get diagonal of A^{-1} efficiently via Cholesky
        L = np.linalg.cholesky(A)
        # Solve L @ L.T @ x = e_i for each i -> use np.linalg.solve
        # More efficient: compute inv_diag using the fact that
        # (A^{-1})_{ii} = 1 / ||L^{-T} e_i||^2
        # Or just use np.linalg.inv for N=200
        A_inv = np.linalg.inv(A)
        diag_inv = np.diag(A_inv)
        y_pred = y - w / diag_inv
        mse = np.mean((y_pred - y)**2)
        return mse, w
    except Exception:
        return float('inf'), None

def _poly_kernel(X1, X2, degree):
    """Polynomial kernel: (1 + X1.X2/d)^degree"""
    # Scale inputs for numerical stability
    X1_norm = X1 / 10.0
    X2_norm = X2 / 10.0
    K = (1.0 + X1_norm @ X2_norm.T / X1_norm.shape[1]) ** degree
    return K

def _matern52_kernel(X1, X2, length_scale):
    """Matern-5/2 kernel."""
    # Compute pairwise distances
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    np.maximum(sq_dists, 0, out=sq_dists)
    d = np.sqrt(sq_dists) / length_scale
    # Matern-5/2: (1 + sqrt(3)*d) * exp(-sqrt(3)*d)
    sqrt3_d = np.sqrt(3.0) * d
    K = (1.0 + sqrt3_d) * np.exp(-sqrt3_d)
    return K

def _rational_fit(X_train, y_train):
    """Fit a rational function: y ≈ (a^T phi(x) + b) / (c^T phi(x) + 1)
    where phi(x) = [x_1, ..., x_d, x_1^2, ..., x_d^2]
    This is a nonlinear fit done via iterative reweighting.
    """
    N, d = X_train.shape
    # Build feature matrix: linear + quadratic terms
    phi = np.hstack([X_train, X_train**2])
    n_phi = phi.shape[1]
    
    # Initialize
    c = np.zeros(n_phi)
    c[-1] = 1.0  # denominator bias
    
    # Iterative fitting
    for it in range(50):
        # Compute current predictions
        num = phi @ np.zeros(n_phi)  # will update
        den = phi @ c + 1.0
        den = np.maximum(den, 1e-6)  # avoid division by zero
        
        # Weighted least squares: minimize sum ((a^T phi_i + b - y_i * (c^T phi_i + 1))^2 / den_i^2)
        # This is equivalent to: minimize sum w_i * (a^T phi_i + b - y_i * den_i)^2
        # where w_i = 1/den_i^2
        
        # Build augmented system
        # We want to solve for [a, b] where a is n_phi and b is scalar
        # phi_aug = [phi, 1]
        phi_aug = np.hstack([phi, np.ones((N, 1))])
        n_aug = phi_aug.shape[1]
        
        # Target: y_i * den_i = y_i * (c^T phi_i + 1)
        target = y_train * den
        
        # Weights
        weights = 1.0 / (den**2)
        
        # Weighted least squares
        W_sqrt = np.sqrt(weights)
        A_w = phi_aug * W_sqrt[:, None]
        b_w = target * W_sqrt
        
        try:
            params, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
        except Exception:
            return None
        
        a = params[:n_phi]
        b = params[n_phi]
        
        # Update c using the numerator: c should minimize sum (a^T phi_i + b - y_i * (c^T phi_i + 1))^2
        # This is another least squares problem for c
        # y_i * c^T phi_i = a^T phi_i + b - y_i
        # c^T phi_i = (a^T phi_i + b) / y_i - 1
        mask = np.abs(y_train) > 1e-6
        if np.sum(mask) < n_phi:
            # Not enough data to solve for c, use ridge
            c_new = np.zeros(n_phi)
            c_new[-1] = 1.0
        else:
            target_c = (phi @ a + b) / y_train[mask] - 1.0
            phi_c = phi[mask]
            try:
                c_new, _, _, _ = np.linalg.lstsq(phi_c, target_c, rcond=None)
            except Exception:
                c_new = np.zeros(n_phi)
                c_new[-1] = 1.0
        
        # Check convergence
        if np.linalg.norm(c_new - c) < 1e-8:
            c = c_new
            break
        c = c_new
    
    # Final predictions
    num = phi @ a + b
    den = phi @ c + 1.0
    den = np.maximum(den, 1e-6)
    y_pred = num / den
    
    # LOOCV-like estimate: use training MSE as proxy (true LOOCV would be expensive)
    mse = np.mean((y_pred - y_train)**2)
    
    def predict(X_test):
        phi_test = np.hstack([X_test, X_test**2])
        num_test = phi_test @ a + b
        den_test = phi_test @ c + 1.0
        den_test = np.maximum(den_test, 1e-6)
        return num_test / den_test
    
    return mse, predict

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 320.0  # 10s safety margin before 330s kill
    
    N, d = X_train.shape
    
    # Pre-compute common quantities
    X_sq = np.sum(X_train**2, axis=1)
    
    candidates = []  # list of (loocv_mse, predict_func)
    
    # --- Model 1: Polynomial Kernel Ridge Regression ---
    poly_degrees = [2, 3, 4]
    for degree in poly_degrees:
        if time.time() > DEADLINE - 30:
            break
        try:
            K = _poly_kernel(X_train, X_train, degree)
            # Normalize kernel
            K = K / np.mean(K)
            
            # Grid search over regularization
            best_mse = float('inf')
            best_w = None
            for lam in np.logspace(-6, 2, 9):
                if time.time() > DEADLINE - 20:
                    break
                mse, w = _loocv_ridge(K, y_train, lam)
                if mse < best_mse:
                    best_mse = mse
                    best_w = w
            
            if best_w is not None:
                def make_poly_predict(degree, w, X_train, X_sq):
                    def predict(X_test):
                        X_test_sq = np.sum(X_test**2, axis=1)
                        K_test = _poly_kernel(X_test, X_train, degree)
                        K_test = K_test / np.mean(_poly_kernel(X_train, X_train, degree))
                        return K_test @ w
                    return predict
                candidates.append((best_mse, make_poly_predict(degree, best_w, X_train, X_sq)))
        except Exception:
            pass
    
    # --- Model 2: Matern-5/2 Gaussian Process ---
    if time.time() < DEADLINE - 60:
        try:
            best_mse = float('inf')
            best_ls = 1.0
            best_var = 1.0
            best_w = None
            
            # Grid search over length_scale and variance
            for ls in np.logspace(-1, 1, 5):
                if time.time() > DEADLINE - 40:
                    break
                for var in np.logspace(-2, 2, 5):
                    if time.time() > DEADLINE - 30:
                        break
                    try:
                        K = _matern52_kernel(X_train, X_train, ls) * var
                        # Add small noise for stability
                        lam = 1e-8
                        mse, w = _loocv_ridge(K, y_train, lam)
                        if mse < best_mse:
                            best_mse = mse
                            best_ls = ls
                            best_var = var
                            best_w = w
                    except Exception:
                        pass
            
            if best_w is not None:
                def make_matern_predict(ls, var, w, X_train, X_sq):
                    def predict(X_test):
                        K_test = _matern52_kernel(X_test, X_train, ls) * var
                        return K_test @ w
                    return predict
                candidates.append((best_mse, make_matern_predict(best_ls, best_var, best_w, X_train, X_sq)))
        except Exception:
            pass
    
    # --- Model 3: Rational Function Approximator ---
    if time.time() < DEADLINE - 30:
        try:
            result = _rational_fit(X_train, y_train)
            if result is not None:
                mse, predict_func = result
                candidates.append((mse, predict_func))
        except Exception:
            pass
    
    # --- Fallback: Simple RBF Kernel Ridge Regression ---
    if not candidates or time.time() < DEADLINE - 20:
        try:
            dist_sq = X_sq[:, None] + X_sq[None, :] - 2 * X_train @ X_train.T
            np.maximum(dist_sq, 0, out=dist_sq)
            
            best_mse = float('inf')
            best_gamma = 1.0
            best_w = None
            
            for gamma in np.logspace(-3, 2, 8):
                if time.time() > DEADLINE - 15:
                    break
                K = np.exp(-gamma * dist_sq)
                for lam in np.logspace(-4, 0, 5):
                    if time.time() > DEADLINE - 10:
                        break
                    mse, w = _loocv_ridge(K, y_train, lam)
                    if mse < best_mse:
                        best_mse = mse
                        best_gamma = gamma
                        best_w = w
            
            if best_w is not None:
                def make_rbf_predict(gamma, w, X_train, X_sq):
                    def predict(X_test):
                        X_test_sq = np.sum(X_test**2, axis=1)
                        dist_sq_test = X_test_sq[:, None] + X_sq[None, :] - 2 * X_test @ X_train.T
                        np.maximum(dist_sq_test, 0, out=dist_sq_test)
                        K_test = np.exp(-gamma * dist_sq_test)
                        return K_test @ w
                    return predict
                candidates.append((best_mse, make_rbf_predict(best_gamma, best_w, X_train, X_sq)))
        except Exception:
            pass
    
    # Select best model
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_predict = candidates[0][1]
    else:
        # Ultimate fallback: mean prediction
        y_mean = np.mean(y_train)
        def predict(X_test):
            return np.full(X_test.shape[0], y_mean)
        best_predict = predict
    
    return best_predict
