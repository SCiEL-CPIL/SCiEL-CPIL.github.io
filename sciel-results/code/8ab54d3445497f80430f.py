# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310  # safety margin before 330s kill


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

    # Quartic cross terms
    if d >= 2:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    for l in range(k, d):
                        if len({i,j,k,l}) == 4:
                            cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
                            names.append(f"x{i}*x{j}*x{k}*x{l}")

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
        cols.append(1.0 / (1.0 + np.exp(-xj)))  # sigmoid
        names.append(f"sigmoid(x{j})")

    # Tanh
    for j in range(d):
        cols.append(np.tanh(X[:, j]))
        names.append(f"tanh(x{j})")
        cols.append(np.tanh(2 * X[:, j]))
        names.append(f"tanh(2x{j})")

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
                cols.append(X[:, i] * np.exp(X[:, j]))
                names.append(f"x{i}*exp(x{j})")
                cols.append(np.exp(X[:, i]) * X[:, j])
                names.append(f"exp(x{i})*x{j}")
                cols.append(X[:, i] * np.sqrt(X[:, j]))
                names.append(f"x{i}*sqrt(x{j})")
                cols.append(np.sqrt(X[:, i]) * X[:, j])
                names.append(f"sqrt(x{i})*x{j}")

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

    # Differences
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] - X[:, j])
                names.append(f"x{i}-x{j}")
                cols.append(X[:, i] * (X[:, i] - X[:, j]))
                names.append(f"x{i}*(x{i}-x{j})")

    return np.column_stack(cols), names


def _fit_lsq(F, y, max_iter=100):
    """Fit y = F @ w using least squares with regularization."""
    n, p = F.shape
    if p == 0:
        return np.zeros(0)
    # Use ridge regression for stability
    lam = 1e-8
    A = F.T @ F + lam * np.eye(p)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    return w


def _orthogonal_matching_pursuit(F, y, max_features=20, tol=1e-8):
    """Fast feature selection using Orthogonal Matching Pursuit (OMP)."""
    n, p = F.shape
    if p <= max_features:
        return list(range(p))
    
    # Center y
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    # Initialize
    residual = y_centered.copy()
    selected = [0]  # start with constant
    F_sel = F[:, selected]
    
    # Compute initial projection
    w = _fit_lsq(F_sel, y)
    pred = F_sel @ w
    residual = y - pred
    
    for _ in range(max_features - 1):
        if _time_left() < 5.0:
            break
            
        # Find feature most correlated with residual
        corr = np.abs(F.T @ residual)
        corr[selected] = 0
        best_idx = np.argmax(corr)
        
        if corr[best_idx] < tol:
            break
            
        selected.append(best_idx)
        F_sel = F[:, selected]
        
        # Refit
        w = _fit_lsq(F_sel, y)
        pred = F_sel @ w
        residual = y - pred
        
        if np.linalg.norm(residual) < 1e-10:
            break
    
    return selected


def _select_features(F, y, names, max_features=20, time_limit=10.0):
    """Select the best subset of features using OMP followed by refinement."""
    n, p = F.shape
    if p <= max_features:
        return list(range(p))

    # Use OMP for fast initial selection
    selected = _orthogonal_matching_pursuit(F, y, max_features=max_features)
    
    F_sel = F[:, selected]
    w = _fit_lsq(F_sel, y)
    pred = F_sel @ w
    best_mse = np.mean((pred - y) ** 2)
    
    # Try to improve with a few greedy additions
    start = time.time()
    improved = True
    while improved and len(selected) < max_features and (time.time() - start) < time_limit:
        improved = False
        best_idx = -1
        best_score = best_mse
        
        # Only try a subset of candidates for speed
        candidates = [j for j in range(p) if j not in selected]
        # Sort by correlation with residual
        residual = y - pred
        corr = np.abs(F[:, candidates].T @ residual)
        top_candidates = [candidates[i] for i in np.argsort(corr)[::-1][:min(20, len(candidates))]]
        
        for j in top_candidates:
            trial = selected + [j]
            F_trial = F[:, trial]
            w_trial = _fit_lsq(F_trial, y)
            pred_trial = F_trial @ w_trial
            mse = np.mean((pred_trial - y) ** 2)
            if mse < best_score * 0.999:  # require meaningful improvement
                best_score = mse
                best_idx = j
                improved = True
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
        F_sel = F[:, selected]
        w = _fit_lsq(F_sel, y)
        pred = F_sel @ w
        best_mse = np.mean((pred - y) ** 2)

    return selected


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

    # Determine how many features to select based on time budget
    time_budget = min(20.0, max(3.0, _time_left() / 10.0))
    max_feat = min(p, 20)

    # Feature selection
    selected = _select_features(F, y_train, names, max_features=max_feat, time_limit=time_budget)

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

    # If fit is poor, try alternative strategies
    if normalized_mse > 0.5 and p > 2:
        # Try with more features
        if len(selected) < max_feat:
            selected2 = _select_features(F, y_train, names, max_features=max_feat, time_limit=min(10.0, _time_left() / 5.0))
            if len(selected2) > len(selected):
                F_sel2 = F[:, selected2]
                w2 = _fit_lsq(F_sel2, y_train)
                pred2 = F_sel2 @ w2
                mse2 = np.mean((pred2 - y_train) ** 2)
                if mse2 < train_mse:
                    F_sel = F_sel2
                    w = w2
                    selected = selected2
                    train_mse = mse2
        
        # Try simpler model as fallback
        F_simple = np.column_stack([np.ones(N)] + [X_train[:, j] for j in range(d)])
        w_simple = _fit_lsq(F_simple, y_train)
        pred_simple = F_simple @ w_simple
        mse_simple = np.mean((pred_simple - y_train) ** 2)
        if mse_simple < train_mse:
            F_sel = F_simple
            w = w_simple
            selected = list(range(F_simple.shape[1]))
            train_mse = mse_simple

    # Final polish: try to improve with local search
    if _time_left() > 10.0 and len(selected) > 2:
        residual = y_train - F_sel @ w
        best_mse = train_mse
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            # Try swapping one feature
            for i in range(1, len(selected)):  # keep constant
                for j in range(p):
                    if j in selected:
                        continue
                    trial = selected.copy()
                    trial[i] = j
                    F_trial = F[:, trial]
                    w_trial = _fit_lsq(F_trial, y_train)
                    pred_trial = F_trial @ w_trial
                    mse_trial = np.mean((pred_trial - y_train) ** 2)
                    if mse_trial < best_mse * 0.99:
                        selected = trial
                        F_sel = F_trial
                        w = w_trial
                        best_mse = mse_trial
                        improved = True
                        break
                if improved:
                    break

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
