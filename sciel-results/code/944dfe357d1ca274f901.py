# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

_START_TIME = time.time()
_DEADLINE = _START_TIME + 315  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _build_basis(X, d):
    """Build a rich set of candidate basis functions evaluated on X."""
    N = X.shape[0]
    cols = [np.ones(N)]
    names = ["1"]

    # Linear
    for j in range(d):
        cols.append(X[:, j])
        names.append(f"x{j}")

    # Squares
    for j in range(d):
        cols.append(X[:, j] ** 2)
        names.append(f"x{j}^2")

    # Cubics
    for j in range(d):
        cols.append(X[:, j] ** 3)
        names.append(f"x{j}^3")

    # Quartics
    for j in range(d):
        cols.append(X[:, j] ** 4)
        names.append(f"x{j}^4")

    # Quintics
    for j in range(d):
        cols.append(X[:, j] ** 5)
        names.append(f"x{j}^5")

    # Pairwise products
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            names.append(f"x{i}*x{j}")

    # Triple products
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
                names.append(f"x{i}*x{j}*x{k}")

    # Quadratic cross terms: x_i^2 * x_j
    for i in range(d):
        for j in range(d):
            if i != j:
                cols.append(X[:, i] ** 2 * X[:, j])
                names.append(f"x{i}^2*x{j}")
                cols.append(X[:, i] * X[:, j] ** 2)
                names.append(f"x{i}*x{j}^2")

    # Trig functions
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        names.append(f"sin(x{j})")
        cols.append(np.cos(X[:, j]))
        names.append(f"cos(x{j})")

    # Exp and log
    for j in range(d):
        xj = X[:, j]
        cols.append(np.exp(xj))
        names.append(f"exp(x{j})")
        cols.append(np.exp(-xj))
        names.append(f"exp(-x{j})")
        cols.append(np.log1p(xj))
        names.append(f"log1p(x{j})")
        cols.append(np.sqrt(xj))
        names.append(f"sqrt(x{j})")

    # Tanh
    for j in range(d):
        cols.append(np.tanh(X[:, j]))
        names.append(f"tanh(x{j})")

    # Cross terms with trig
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i]) * X[:, j])
                names.append(f"sin(x{i})*x{j}")
                cols.append(np.cos(X[:, i]) * X[:, j])
                names.append(f"cos(x{i})*x{j}")
                cols.append(np.sin(X[:, i]) * np.sin(X[:, j]))
                names.append(f"sin(x{i})*sin(x{j})")
                cols.append(np.cos(X[:, i]) * np.cos(X[:, j]))
                names.append(f"cos(x{i})*cos(x{j})")
                # Add missing mixed trig
                cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
                names.append(f"sin(x{i})*cos(x{j})")
                cols.append(np.cos(X[:, i]) * np.sin(X[:, j]))
                names.append(f"cos(x{i})*sin(x{j})")
                
                cols.append(X[:, i] * np.exp(X[:, j]))
                names.append(f"x{i}*exp(x{j})")
                cols.append(np.exp(X[:, i]) * X[:, j])
                names.append(f"exp(x{i})*x{j}")
                
                # Exp-Exp interaction
                cols.append(np.exp(X[:, i]) * np.exp(X[:, j]))
                names.append(f"exp(x{i})*exp(x{j})")

    # Sum and product of all inputs
    cols.append(np.sum(X, axis=1))
    names.append("sum(x)")
    cols.append(np.prod(X, axis=1))
    names.append("prod(x)")
    cols.append(np.sum(X ** 2, axis=1))
    names.append("sum(x^2)")

    # Distance from origin
    cols.append(np.linalg.norm(X, axis=1))
    names.append("norm(x)")

    return np.column_stack(cols), names


def _fit_lsq(F, y, max_iter=100):
    """Fit y = F @ w using least squares with regularization."""
    n, p = F.shape
    if p == 0:
        return np.zeros(0)
    # Use ridge regression for stability
    lam = 1e-6
    A = F.T @ F + lam * np.eye(p)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    return w


def _lasso_cd(F, y, lam=0.1, alpha=0.0, max_iter=500, tol=1e-4):
    """
    Simple Coordinate Descent Lasso/Elastic Net solver.
    Minimizes: 0.5/N * ||y - Fw||^2 + lam * ||w||_1 + 0.5 * alpha * ||w||^2
    """
    n, p = F.shape
    w = np.zeros(p)
    
    # Precompute column norms squared: ||f_j||^2
    col_norms_sq = np.sum(F ** 2, axis=0)
    
    # Precompute F.T @ y
    Fty = F.T @ y
    
    # Residual r = y - F @ w
    r = y.copy()
    
    for _ in range(max_iter):
        max_diff = 0.0
        for j in range(p):
            # Update residual to exclude j-th term: r += F_j * w_j
            r += F[:, j] * w[j]
            
            # rho_j = (1/N) * F_j^T * r
            rho_j = (F[:, j] @ r) / n
            
            # Soft threshold
            # w_j_new = S(rho_j, lam) / ( (1/N)*||F_j||^2 + alpha )
            denom = col_norms_sq[j] / n + alpha
            
            if denom == 0:
                w_new = 0.0
            else:
                if rho_j > lam:
                    w_new = (rho_j - lam) / denom
                elif rho_j < -lam:
                    w_new = (rho_j + lam) / denom
                else:
                    w_new = 0.0
            
            # Update residual: r -= F_j * (w_new - w_j)
            delta_w = w_new - w[j]
            r -= F[:, j] * delta_w
            max_diff = max(max_diff, abs(delta_w))
            w[j] = w_new
            
        if max_diff < tol:
            break
            
    return w


def _select_features_lasso(F, y, names, time_limit=10.0):
    """
    Use Lasso to select features.
    Returns the indices of non-zero coefficients.
    """
    n, p = F.shape
    
    # Try a range of lambda values
    # Scale lambda based on data variance
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
        
    # F_max = max_j |F_j^T y| / N
    Fty = F.T @ y
    F_max = np.max(np.abs(Fty)) / n
    
    # Log-spaced lambda values
    lambdas = np.logspace(np.log10(F_max * 1e-3), np.log10(F_max * 1e-1), 10)
    
    best_w = None
    best_mse = np.inf
    best_idx = None
    
    start_time = time.time()
    
    for lam in lambdas:
        if time.time() - start_time > time_limit:
            break
            
        w = _lasso_cd(F, y, lam=lam, alpha=1e-4, max_iter=200, tol=1e-5)
        
        # Calculate MSE
        pred = F @ w
        mse = np.mean((pred - y) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_w = w
            # Get indices where w is significantly non-zero
            # Use a threshold relative to the max coefficient
            if np.max(np.abs(w)) > 1e-6:
                threshold = 1e-6 * np.max(np.abs(w))
                best_idx = np.where(np.abs(w) > threshold)[0]
            else:
                best_idx = np.array([0]) # Fallback to constant
    
    if best_idx is None:
        best_idx = np.array([0])
        
    return best_idx


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape

    # Handle edge cases
    if N == 0:
        return lambda X: np.zeros(len(X))

    # Build rich basis
    F, names = _build_basis(X_train, d)

    # Remove non-finite columns
    finite_cols = np.all(np.isfinite(F), axis=0)
    F = F[:, finite_cols]
    names = [n for n, f in zip(names, finite_cols) if f]
    p = F.shape[1]

    if p == 0:
        return lambda X: np.zeros(len(X))

    # Determine time budget
    time_budget = min(20.0, max(5.0, _time_left() / 10.0))
    
    # Use Lasso for feature selection
    selected = _select_features_lasso(F, y_train, names, time_limit=time_budget)
    
    # Ensure constant term (index 0) is included if possible
    if 0 not in selected:
        selected = np.append(selected, 0)
        
    F_sel = F[:, selected]
    
    # Refit with Ridge on selected features to get optimal weights
    w = _fit_lsq(F_sel, y_train)
    
    # Verify fit quality
    pred_train = F_sel @ w
    train_mse = np.mean((pred_train - y_train) ** 2)
    y_var = np.var(y_train)
    if y_var > 0:
        normalized_mse = train_mse / y_var
    else:
        normalized_mse = train_mse

    # Fallback: If Lasso selection is bad, try simple linear
    if normalized_mse > 1.0 and p > 2:
        F_simple = np.column_stack([np.ones(N)] + [X_train[:, j] for j in range(d)])
        w_simple = _fit_lsq(F_simple, y_train)
        pred_simple = F_simple @ w_simple
        mse_simple = np.mean((pred_simple - y_train) ** 2)
        if mse_simple < train_mse:
            F_sel = F_simple
            w = w_simple
            selected = list(range(F_simple.shape[1]))

    sel_indices = selected

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F_pred, _ = _build_basis(X, d)
        # Apply same finite column mask
        F_pred = F_pred[:, finite_cols]
        F_pred_sel = F_pred[:, sel_indices]
        return F_pred_sel @ w

    return predict
# EVOLVE-BLOCK-END
