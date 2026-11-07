"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

_DEADLINE = time.time() + 320  # safety margin before 330s wall-clock kill


def _elapsed():
    return time.time() - _DEADLINE  # positive when over


def _build_rich_basis(X):
    """Build a rich feature basis: polynomials up to degree 3, trig, exp, log, tanh."""
    d = X.shape[1]
    n = X.shape[0]
    cols = [np.ones(n)]
    
    # Linear terms
    for j in range(d):
        cols.append(X[:, j])
    
    # Quadratic terms
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    
    # Cubic terms
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    
    # Trigonometric terms (for each input)
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
    
    # Exponential terms (scaled to avoid overflow)
    for j in range(d):
        x_scaled = X[:, j] / 5.0  # scale based on max range
        cols.append(np.exp(x_scaled))
    
    # Log terms
    for j in range(d):
        cols.append(np.log(X[:, j] + 1e-10))
    
    # Tanh terms
    for j in range(d):
        cols.append(np.tanh(X[:, j]))
    
    # Interactions with trig
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * np.sin(X[:, j]))
            cols.append(X[:, i] * np.cos(X[:, j]))
    
    return np.column_stack(cols)


def _fit_ridge(F, y, alpha):
    """Fit ridge regression and return coefficients and training MSE."""
    n_features = F.shape[1]
    A = F.T @ F + alpha * np.eye(n_features)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    pred = F @ w
    mse = np.mean((pred - y) ** 2)
    return w, mse


def _forward_selection(F, y, max_features=15, alpha=1e-4, time_budget=25):
    """Forward feature selection to find sparse model."""
    n_samples, n_features = F.shape
    selected = []
    best_mse = np.inf
    start_time = time.time()
    
    # Start with intercept (index 0)
    selected.append(0)
    
    while len(selected) < max_features:
        if time.time() - start_time > time_budget:
            break
        
        best_idx = -1
        best_mse_iter = best_mse
        
        for idx in range(n_features):
            if idx in selected:
                continue
            
            trial_selected = selected + [idx]
            F_trial = F[:, trial_selected]
            w, mse = _fit_ridge(F_trial, y, alpha)
            
            if mse < best_mse_iter:
                best_mse_iter = mse
                best_idx = idx
        
        if best_idx == -1 or best_mse_iter >= best_mse:
            break
        
        selected.append(best_idx)
        best_mse = best_mse_iter
    
    F_final = F[:, selected]
    w_final, mse_final = _fit_ridge(F_final, y, alpha)
    return selected, w_final, mse_final


def fit(X_train, y_train):
    d = X_train.shape[1]
    
    # Build rich basis
    F_train = _build_rich_basis(X_train)
    
    # Forward selection for sparse model
    try:
        selected_idx, w_ridge, mse_fs = _forward_selection(
            F_train, y_train, max_features=15, alpha=1e-4, time_budget=20
        )
    except Exception:
        selected_idx = [0]
        w_ridge = np.zeros(1)
        mse_fs = np.mean(y_train ** 2)
    
    # Also try simple polynomial ridge as fallback/comparison
    try:
        F_poly = _build_rich_basis(X_train)[:, :10]  # first 10 features (intercept + linear)
        w_poly, mse_poly = _fit_ridge(F_poly, y_train, 1e-6)
    except Exception:
        w_poly = None
        mse_poly = np.inf
    
    # Choose best model
    if mse_fs <= mse_poly:
        final_selected = selected_idx
        final_w = w_ridge
        final_mse = mse_fs
    else:
        final_selected = list(range(min(10, F_train.shape[1])))
        final_w = w_poly
        final_mse = mse_poly
    
    # Store the selected indices and weights
    selected_indices = final_selected
    weights = final_w
    
    def predict(X):
        F = _build_rich_basis(X)
        F_sel = F[:, selected_indices]
        return F_sel @ weights
    
    return predict
# EVOLVE-BLOCK-END
