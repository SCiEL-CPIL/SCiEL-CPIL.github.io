# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

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
        cols.append(np.sin(2 * X[:, j]))
        names.append(f"sin(2x{j})")
        cols.append(np.cos(2 * X[:, j]))
        names.append(f"cos(2x{j})")

    # Exp and log (safe versions)
    for j in range(d):
        xj = X[:, j]
        # Clip to avoid overflow
        xj_clipped = np.clip(xj, -50, 50)
        cols.append(np.exp(xj_clipped))
        names.append(f"exp(x{j})")
        cols.append(np.exp(-xj_clipped))
        names.append(f"exp(-x{j})")
        # Safe log
        xj_pos = np.maximum(xj, 1e-10)
        cols.append(np.log(xj_pos))
        names.append(f"log(x{j})")
        cols.append(np.log1p(xj))
        names.append(f"log1p(x{j})")
        cols.append(np.sqrt(xj))
        names.append(f"sqrt(x{j})")

    # Tanh
    for j in range(d):
        cols.append(np.tanh(X[:, j]))
        names.append(f"tanh(x{j})")
        cols.append(np.arctan(X[:, j]))
        names.append(f"arctan(x{j})")

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
                cols.append(X[:, i] * np.exp(np.clip(X[:, j], -50, 50)))
                names.append(f"x{i}*exp(x{j})")
                cols.append(np.exp(np.clip(X[:, i], -50, 50)) * X[:, j])
                names.append(f"exp(x{i})*x{j}")
                cols.append(X[:, i] / (X[:, j] + 1e-10))
                names.append(f"x{i}/x{j}")
                cols.append(X[:, j] / (X[:, i] + 1e-10))
                names.append(f"x{j}/x{i}")

    # Sum and product of all inputs
    cols.append(np.sum(X, axis=1))
    names.append("sum(x)")
    cols.append(np.prod(X, axis=1))
    names.append("prod(x)")
    cols.append(np.sum(X ** 2, axis=1))
    names.append("sum(x^2)")
    cols.append(np.sum(X ** 3, axis=1))
    names.append("sum(x^3)")

    # Distance from origin
    cols.append(np.linalg.norm(X, axis=1))
    names.append("norm(x)")
    cols.append(np.linalg.norm(X, axis=1) ** 2)
    names.append("norm(x)^2")

    # Min and max
    cols.append(np.min(X, axis=1))
    names.append("min(x)")
    cols.append(np.max(X, axis=1))
    names.append("max(x)")

    return np.column_stack(cols), names


def _fit_lsq(F, y, lam=1e-6):
    """Fit y = F @ w using ridge regression."""
    n, p = F.shape
    if p == 0:
        return np.zeros(0)
    A = F.T @ F + lam * np.eye(p)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    return w


def _fit_lasso(F, y, alphas=None, max_iter=5000, tol=1e-6):
    """Fit y = F @ w using coordinate descent Lasso."""
    n, p = F.shape
    if p == 0:
        return np.zeros(0)
    
    if alphas is None:
        alphas = np.logspace(-4, 1, 10)
    
    best_w = np.zeros(p)
    best_mse = np.inf
    
    for alpha in alphas:
        w = np.zeros(p)
        for _ in range(max_iter):
            w_old = w.copy()
            for j in range(p):
                # Compute residual excluding feature j
                r = y - F @ w + F[:, j] * w[j]
                # Compute z_j = F_j^T r / n
                z_j = np.dot(F[:, j], r) / n
                # Soft thresholding
                if z_j > alpha:
                    w_new_j = (z_j - alpha) / (np.dot(F[:, j], F[:, j]) / n)
                elif z_j < -alpha:
                    w_new_j = (z_j + alpha) / (np.dot(F[:, j], F[:, j]) / n)
                else:
                    w_new_j = 0.0
                
                w[j] = w_new_j
            
            # Check convergence
            if np.max(np.abs(w - w_old)) < tol:
                break
        
        pred = F @ w
        mse = np.mean((pred - y) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_w = w.copy()
    
    return best_w


def _select_features_l1(F, y, names, max_features=15, time_limit=10.0):
    """Select features using L1 regularization."""
    n, p = F.shape
    if p <= max_features:
        return list(range(p))
    
    start = time.time()
    
    # Try a range of alpha values
    alphas = np.logspace(-5, 0, 15)
    best_w = np.zeros(p)
    best_mse = np.inf
    best_selected = [0]
    
    for alpha in alphas:
        if time.time() - start > time_limit:
            break
        
        w = _fit_lasso(F, y, alphas=[alpha], max_iter=3000)
        pred = F @ w
        mse = np.mean((pred - y) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_w = w.copy()
            # Get selected features (non-zero weights)
            selected = [j for j in range(p) if abs(w[j]) > 1e-8]
            if 0 not in selected:
                selected.append(0)  # Always include constant
            best_selected = selected
    
    # If too many features selected, keep only top max_features
    if len(best_selected) > max_features:
        # Sort by absolute weight magnitude
        weights_sorted = np.argsort(np.abs(best_w))[::-1]
        best_selected = [0]  # Always include constant
        for j in weights_sorted:
            if j not in best_selected and len(best_selected) < max_features:
                best_selected.append(j)
    
    return best_selected


def _polish_fit(F, y, w_init, max_iter=1000, tol=1e-10):
    """High-precision polish using scipy minimize on MSE."""
    n, p = F.shape
    
    def mse_objective(w):
        pred = F @ w
        return np.mean((pred - y) ** 2)
    
    try:
        result = minimize(
            mse_objective,
            w_init,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
        )
        if result.success:
            return result.x
    except Exception:
        pass
    
    return w_init


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
    max_feat = min(p, 15)

    # Feature selection using L1 regularization
    selected = _select_features_l1(F, y_train, names, max_features=max_feat, time_limit=time_budget)

    F_sel = F[:, selected]
    w = _fit_lsq(F_sel, y_train)

    # Verify fit quality
    pred_train = F_sel @ w
    train_mse = np.mean((pred_train - y_train) ** 2)
    y_var = np.var(y_train)
    if y_var > 0:
        normalized_mse = train_mse / y_var
    else:
        normalized_mse = train_mse

    # If fit is poor, try simpler models
    if normalized_mse > 1.0 and p > 2:
        # Try just linear + constant
        F_simple = np.column_stack([np.ones(N)] + [X_train[:, j] for j in range(d)])
        w_simple = _fit_lsq(F_simple, y_train)
        pred_simple = F_simple @ w_simple
        mse_simple = np.mean((pred_simple - y_train) ** 2)
        if mse_simple < train_mse:
            F_sel = F_simple
            w = w_simple
            selected = list(range(F_simple.shape[1]))
            train_mse = mse_simple
    
    # Final polish with high-precision optimization
    if _time_left() > 5.0:
        w = _polish_fit(F_sel, y_train, w, max_iter=500, tol=1e-12)

    # Capture the selected feature indices and basis for prediction
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
