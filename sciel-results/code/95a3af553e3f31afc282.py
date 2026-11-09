"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def fit(X_train, y_train):
    """Fit using high-order polynomial regression for 1D data,
    with a fallback for multi-dimensional inputs."""
    t0 = time.time()
    deadline = 325.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    if d == 1:
        x = X[:, 0].copy()
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # Try polynomial fits of increasing degree, use the best one that doesn't overfit
        # For noiseless data, a high-degree polynomial can interpolate exactly
        # but may oscillate. We use a moderate degree with regularization.
        
        # First, try to determine a good degree by checking residuals
        # For noiseless data, we want exact interpolation but stable extrapolation
        
        best_degree = 10
        best_w = None
        best_residual = np.inf
        
        # Try degrees from 3 to 20, pick the one with best validation score
        # Use a simple train/test split on the sorted data
        n_train = int(0.8 * n)
        
        for deg in range(3, 21):
            if time.time() > deadline:
                break
                
            # Build Vandermonde matrix
            V_train = np.vander(x_sorted[:n_train], N=deg+1, increasing=True)
            V_test = np.vander(x_sorted[n_train:], N=deg+1, increasing=True)
            
            try:
                # Use ridge regression for stability
                lam = 1e-8 * deg
                A = V_train.T @ V_train + lam * np.eye(deg + 1)
                b = V_train.T @ y_sorted[:n_train]
                w = np.linalg.solve(A, b)
                
                y_pred_test = V_test @ w
                residual = np.mean((y_pred_test - y_sorted[n_train:])**2)
                
                if residual < best_residual:
                    best_residual = residual
                    best_degree = deg
                    best_w = w
            except:
                continue
        
        # If we didn't find a good degree, use a default
        if best_w is None:
            best_degree = 10
            V_train = np.vander(x_sorted, N=best_degree+1, increasing=True)
            lam = 1e-8 * best_degree
            A = V_train.T @ V_train + lam * np.eye(best_degree + 1)
            b = V_train.T @ y_sorted
            best_w = np.linalg.solve(A, b)
        else:
            # Refit on all data with the best degree
            V_all = np.vander(x_sorted, N=best_degree+1, increasing=True)
            lam = 1e-8 * best_degree
            A = V_all.T @ V_all + lam * np.eye(best_degree + 1)
            b = V_all.T @ y_sorted
            best_w = np.linalg.solve(A, b)

        def predict_poly(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            xt = X_test[:, 0]
            V_test = np.vander(xt, N=best_degree+1, increasing=True)
            return V_test @ best_w

        return predict_poly

    else:
        # Multi-dimensional fallback: polynomial regression
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)

        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)

        def predict_poly(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, d)
            return features(X_test) @ w

        return predict_poly

# EVOLVE-BLOCK-END
