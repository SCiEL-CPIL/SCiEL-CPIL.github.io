"""
Symbolic regression using a rich global basis (polynomial, trigonometric, exponential,
logarithmic, tanh, abs) with adaptive ridge regression.
This is a fundamentally different approach from local LOESS: it fits a single global
model across the entire domain, which can capture smooth functions more effectively
when the function has a clear global structure.
"""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0
_t_start = time.time()


def _time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _t_start)


def _build_rich_features(X):
    """
    Build a rich feature matrix from input X.
    X: (N,) or (N, 1) array
    Returns: (N, P) feature matrix
    """
    X = np.asarray(X, dtype=np.float64).flatten()
    n = len(X)
    
    # Normalize X to [-1, 1] for numerical stability in basis functions
    x_min, x_max = X.min(), X.max()
    if x_max - x_min < 1e-10:
        Xn = np.zeros(n)
    else:
        Xn = 2.0 * (X - x_min) / (x_max - x_min) - 1.0
    
    features = [np.ones(n)]
    
    # Polynomial terms up to degree 8
    for d in range(1, 9):
        features.append(Xn ** d)
    
    # Trigonometric terms
    for k in range(1, 6):
        features.append(np.sin(k * Xn))
        features.append(np.cos(k * Xn))
    
    # Exponential terms
    for a in [0.5, 1.0, 2.0]:
        features.append(np.exp(a * Xn))
        features.append(np.exp(-a * Xn))
    
    # Logarithmic terms (safe version)
    eps = 1e-8
    features.append(np.log1p(np.abs(Xn) + eps))
    features.append(np.log1p(np.abs(Xn + 1.0) + eps))
    features.append(np.log1p(np.abs(Xn - 1.0) + eps))
    
    # Tanh terms
    for a in [0.5, 1.0, 2.0]:
        features.append(np.tanh(a * Xn))
    
    # Absolute value
    features.append(np.abs(Xn))
    
    # Sigmoid-like
    features.append(1.0 / (1.0 + np.exp(-2.0 * Xn)))
    
    # Piecewise linear-like (using abs)
    for c in [-0.5, 0.0, 0.5]:
        features.append(np.maximum(0.0, Xn - c))
        features.append(np.maximum(0.0, c - Xn))
    
    F = np.column_stack(features)
    
    # Return F and the normalization parameters
    return F, Xn, x_min, x_max


def _fit_ridge(y, F, lambdas):
    """
    Fit ridge regression for multiple lambda values.
    Returns: (best_lambda, best_beta, best_cv_mse)
    """
    n, p = F.shape
    best_lambda = lambdas[0]
    best_beta = None
    best_mse = np.inf
    
    for lam in lambdas:
        # Ridge: beta = (F^T F + lam * I)^{-1} F^T y
        try:
            A = F.T @ F + lam * np.eye(p)
            b = F.T @ y
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        
        # Compute training MSE (as proxy; true CV would be better but slower)
        preds = F @ beta
        mse = np.mean((preds - y) ** 2)
        
        # Add a small penalty for large norms to encourage simpler models
        penalty = 0.01 * np.sum(beta ** 2)
        total = mse + penalty
        
        if total < best_mse:
            best_mse = total
            best_lambda = lam
            best_beta = beta
    
    return best_lambda, best_beta, best_mse


def fit(X_train, y_train):
    global _t_start
    _t_start = time.time()
    
    X = np.asarray(X_train, dtype=np.float64).flatten()
    y = np.asarray(y_train, dtype=np.float64).flatten()
    
    n = len(X)
    
    # Build rich feature matrix
    F, Xn, x_min, x_max = _build_rich_features(X)
    
    # Try a range of regularization strengths
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    best_lambda, best_beta, best_mse = _fit_ridge(y, F, lambdas)
    
    # If time permits, do a finer search around best_lambda
    if _time_left() > 5.0:
        fine_lambdas = np.logspace(np.log10(best_lambda / 3), np.log10(best_lambda * 3), 5)
        fine_lambdas = np.unique(np.concatenate([lambdas, fine_lambdas]))
        best_lambda, best_beta, best_mse = _fit_ridge(y, F, fine_lambdas)
    
    # Final prediction function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64).flatten()
        
        # Use the same normalization as training
        if x_max - x_min < 1e-10:
            Xn_test = np.zeros(len(X_test))
        else:
            Xn_test = 2.0 * (X_test - x_min) / (x_max - x_min) - 1.0
        
        # Build features for test points
        F_test = []
        F_test.append(np.ones(len(X_test)))
        for d in range(1, 9):
            F_test.append(Xn_test ** d)
        for k in range(1, 6):
            F_test.append(np.sin(k * Xn_test))
            F_test.append(np.cos(k * Xn_test))
        for a in [0.5, 1.0, 2.0]:
            F_test.append(np.exp(a * Xn_test))
            F_test.append(np.exp(-a * Xn_test))
        eps = 1e-8
        F_test.append(np.log1p(np.abs(Xn_test) + eps))
        F_test.append(np.log1p(np.abs(Xn_test + 1.0) + eps))
        F_test.append(np.log1p(np.abs(Xn_test - 1.0) + eps))
        for a in [0.5, 1.0, 2.0]:
            F_test.append(np.tanh(a * Xn_test))
        F_test.append(np.abs(Xn_test))
        F_test.append(1.0 / (1.0 + np.exp(-2.0 * Xn_test)))
        for c in [-0.5, 0.0, 0.5]:
            F_test.append(np.maximum(0.0, Xn_test - c))
            F_test.append(np.maximum(0.0, c - Xn_test))
        
        F_test = np.column_stack(F_test)
        
        preds = F_test @ best_beta
        
        # Ensure finite predictions
        preds = np.nan_to_num(preds, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return preds
    
    return predict
