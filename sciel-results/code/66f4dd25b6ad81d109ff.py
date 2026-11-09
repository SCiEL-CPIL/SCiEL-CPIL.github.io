import numpy as np
import time
from scipy.linalg import solve

def _fit_poly(X, y, deg):
    """Fit a polynomial of degree deg."""
    X = X.flatten()
    try:
        coeffs = np.polyfit(X, y, deg)
        # Convert to Horner's method or use np.polyval
        def predict(x):
            x = np.asarray(x).flatten()
            return np.polyval(coeffs, x)
        # Calculate train score
        y_pred = predict(X)
        mse = np.mean((y_pred - y)**2)
        var_y = np.var(y)
        if var_y < 1e-10:
            var_y = 1.0
        score = mse / var_y
        return predict, score
    except Exception:
        return None, np.inf

def _fit_fourier(X, y, n_terms):
    """Fit a Fourier series."""
    X = X.flatten()
    # Normalize X to [0, 2*pi] or use raw? 
    # Raw X is better if we adjust frequencies.
    # Let's use frequencies k * (2*pi / (max(X)-min(X)))
    x_min = np.min(X)
    x_max = np.max(X)
    if x_max - x_min < 1e-8:
        return None, np.inf
    period = x_max - x_min
    
    # Design matrix
    # Columns: 1, sin(2*pi*k*x/period), cos(2*pi*k*x/period)
    cols = [np.ones_like(X)]
    for k in range(1, n_terms + 1):
        angle = 2 * np.pi * k * X / period
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    
    A = np.column_stack(cols)
    
    try:
        # Ridge regression for stability
        lam = 1e-6
        coeffs, _ = np.linalg.lstsq(A, y, rcond=None)
        # Actually, let's use ridge explicitly
        # (A^T A + lam*I) c = A^T y
        # But lstsq is fine if we add small regularization to A? 
        # Let's just use lstsq, it's robust enough for 200 points.
        
        def predict(x):
            x = np.asarray(x).flatten()
            cols = [np.ones_like(x)]
            for k in range(1, n_terms + 1):
                angle = 2 * np.pi * k * x / period
                cols.append(np.sin(angle))
                cols.append(np.cos(angle))
            A_test = np.column_stack(cols)
            return A_test @ coeffs
        
        y_pred = predict(X)
        mse = np.mean((y_pred - y)**2)
        var_y = np.var(y)
        if var_y < 1e-10:
            var_y = 1.0
        score = mse / var_y
        return predict, score
    except Exception:
        return None, np.inf

def _fit_rbf(X, y, width):
    """Fit Radial Basis Function model."""
    X = X.flatten()
    centers = X # Use all training points as centers
    
    # Kernel matrix K
    # K[i,j] = exp(- (X[i] - X[j])^2 / (2 * width^2))
    diff = X[:, np.newaxis] - X[np.newaxis, :]
    K = np.exp(-diff**2 / (2 * width**2))
    
    # Ridge regularization
    lam = 1e-3
    try:
        coeffs = np.linalg.solve(K + lam * np.eye(len(X)), y)
        
        def predict(x):
            x = np.asarray(x).flatten()
            diff = x[:, np.newaxis] - centers[np.newaxis, :]
            K_test = np.exp(-diff**2 / (2 * width**2))
            return K_test @ coeffs
        
        y_pred = predict(X)
        mse = np.mean((y_pred - y)**2)
        var_y = np.var(y)
        if var_y < 1e-10:
            var_y = 1.0
        score = mse / var_y
        return predict, score
    except Exception:
        return None, np.inf

def fit(X_train, y_train):
    start_time = time.time()
    X = np.asarray(X_train).flatten()
    y = np.asarray(y_train).flatten()
    
    best_predict = None
    best_score = np.inf
    
    # Candidates
    # Polynomials
    for deg in [2, 3, 4, 5, 6, 8, 10]:
        pred, score = _fit_poly(X, y, deg)
        if pred is not None and score < best_score:
            best_score = score
            best_predict = pred
        if time.time() - start_time > 290:
            break
            
    # Fourier
    for n_terms in [5, 10, 20, 30, 50]:
        pred, score = _fit_fourier(X, y, n_terms)
        if pred is not None and score < best_score:
            best_score = score
            best_predict = pred
        if time.time() - start_time > 290:
            break
            
    # RBF
    for width in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        pred, score = _fit_rbf(X, y, width)
        if pred is not None and score < best_score:
            best_score = score
            best_predict = pred
        if time.time() - start_time > 290:
            break
            
    # Fallback: Mean
    if best_predict is None:
        mean_y = np.mean(y)
        def predict(x):
            x = np.asarray(x)
            if x.ndim == 1:
                return np.full(x.shape, mean_y)
            else:
                return np.full(x.shape[0], mean_y)
        best_predict = predict

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            return best_predict(X_test)
        else:
            # Handle 2D input (N, 1)
            return best_predict(X_test[:, 0])
            
    return predict
