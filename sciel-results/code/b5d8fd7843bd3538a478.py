"""Symbolic regression via adaptive basis expansion with cross-validation."""
import time
import numpy as np
import sys

def fit(X_train, y_train):
    """
    Fit a regression model and return a predict callable.
    Uses an adaptive linear basis expansion with Ridge Regression and
    Cross-Validation to handle diverse functional forms robustly.
    """
    start_time = time.time()
    deadline = start_time + 285  # 15s safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0]

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    # Generate a diverse set of basis functions
    def generate_basis_functions(x_vals):
        """Generate a diverse set of basis functions evaluated at x_vals."""
        x_safe = np.where(np.abs(x_vals) < 1e-10, 1e-10, x_vals)
        x_pos = np.where(x_vals > 0, x_vals, 1e-10)
        
        cols = [np.ones(len(x_vals))]
        
        # Polynomials up to degree 10
        for k in range(1, 11):
            cols.append(x_vals ** k)
        
        # Fourier terms
        for k in range(1, 15):
            cols.append(np.sin(k * x_vals))
            cols.append(np.cos(k * x_vals))
        
        # Exponential terms
        for k in [0.25, 0.5, 1.0, 2.0, 4.0]:
            cols.append(np.exp(k * x_vals))
            cols.append(np.exp(-k * x_vals))
        
        # Log terms (for positive x)
        cols.append(np.log(x_pos))
        cols.append(np.log1p(np.abs(x_vals)))
        
        # Reciprocal terms
        cols.append(1.0 / x_safe)
        cols.append(1.0 / (x_safe ** 2))
        
        # Hyperbolic
        cols.append(np.tanh(x_vals))
        cols.append(np.tanh(2 * x_vals))
        cols.append(np.tanh(0.5 * x_vals))
        
        # Sigmoid-like
        cols.append(1.0 / (1.0 + np.exp(-x_vals)))
        cols.append(1.0 / (1.0 + np.exp(-2 * x_vals)))
        
        # Absolute value and sign
        cols.append(np.abs(x_vals))
        cols.append(np.sign(x_vals))
        
        # Square root
        cols.append(np.sqrt(np.abs(x_vals)))
        
        # Gaussian bumps at various centers
        x_min, x_max = np.min(x_vals), np.max(x_vals)
        x_range = x_max - x_min
        if x_range > 1e-10:
            for frac in [0.25, 0.5, 0.75]:
                center = x_min + frac * x_range
                width = x_range / 4
                cols.append(np.exp(-((x_vals - center) / width) ** 2))
        
        # RBF-like features
        for center in np.quantile(x_vals, [0.2, 0.4, 0.6, 0.8]):
            for bw in [0.5, 1.0, 2.0]:
                cols.append(np.exp(-((x_vals - center) / bw) ** 2))
        
        return np.column_stack(cols)

    # Cross-validation with ridge regression
    def cv_ridge(A, y, alpha_values, k=5):
        """Perform k-fold CV for ridge regression, return best alpha and score."""
        if elapsed() > deadline - 3:
            return None, np.inf
        
        indices = np.arange(len(y))
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        
        best_alpha = alpha_values[0]
        best_score = np.inf
        
        for alpha in alpha_values:
            errors = []
            for i in range(k):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                
                try:
                    A_tr = A[tr_idx]
                    y_tr = y[tr_idx]
                    A_val = A[val_idx]
                    y_val = y[val_idx]
                    
                    # Ridge regression: (A^T A + alpha * I) w = A^T y
                    n_feat = A_tr.shape[1]
                    reg_matrix = A_tr.T @ A_tr + alpha * np.eye(n_feat)
                    rhs = A_tr.T @ y_tr
                    
                    try:
                        w = np.linalg.solve(reg_matrix, rhs)
                    except np.linalg.LinAlgError:
                        w = np.linalg.lstsq(reg_matrix, rhs, rcond=None)[0]
                    
                    if not np.all(np.isfinite(w)):
                        return None, np.inf
                    
                    pred_val = A_val @ w
                    if not np.all(np.isfinite(pred_val)):
                        return None, np.inf
                    
                    mse = np.mean((pred_val - y_val) ** 2)
                    errors.append(mse)
                except Exception:
                    return None, np.inf
            
            mean_err = np.mean(errors)
            if mean_err < best_score:
                best_score = mean_err
                best_alpha = alpha
        
        return best_alpha, best_score

    # Build basis and fit
    A_full = generate_basis_functions(x)
    
    # Check for non-finite values in basis
    if not np.all(np.isfinite(A_full)):
        A_full = np.where(np.isfinite(A_full), A_full, 0.0)
    
    # Try multiple alpha values
    alpha_values = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    best_alpha, best_cv_score = cv_ridge(A_full, y, alpha_values)
    
    # If CV failed, fall back to simple polynomial
    if best_alpha is None:
        # Simple linear regression as fallback
        A_simple = np.column_stack([np.ones(n), x])
        w, *_ = np.linalg.lstsq(A_simple, y, rcond=None)
        if not np.all(np.isfinite(w)):
            mean_y = np.mean(y)
            def predict(X_in):
                return np.full(len(X_in), mean_y)
            return predict
        
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            Ai = np.column_stack([np.ones(len(xi)), xi])
            res = Ai @ w
            res = np.where(np.isfinite(res), res, mean_y)
            return res
        return predict

    # Fit with best alpha on all data
    n_feat = A_full.shape[1]
    reg_matrix = A_full.T @ A_full + best_alpha * np.eye(n_feat)
    rhs = A_full.T @ y
    
    try:
        w = np.linalg.solve(reg_matrix, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(reg_matrix, rhs, rcond=None)[0]
    
    if not np.all(np.isfinite(w)):
        # Fallback to mean
        mean_y = np.mean(y)
        def predict(X_in):
            return np.full(len(X_in), mean_y)
        return predict

    # Create predict function
    def predict(X_in):
        X_in = np.asarray(X_in, dtype=np.float64)
        xi = X_in[:, 0]
        try:
            A_pred = generate_basis_functions(xi)
            if not np.all(np.isfinite(A_pred)):
                A_pred = np.where(np.isfinite(A_pred), A_pred, 0.0)
            res = A_pred @ w
            res = np.asarray(res, dtype=np.float64)
            if res.shape != (len(X_in),):
                res = np.atleast_1d(res).flatten()
                if len(res) != len(X_in):
                    res = np.resize(res, len(X_in))
            if not np.all(np.isfinite(res)):
                res = np.where(np.isfinite(res), res, np.mean(y))
            return res
        except Exception:
            return np.full(len(X_in), np.mean(y))

    return predict
