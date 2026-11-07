"""Symbolic regression over 10 hidden equations.
Combines a rich feature basis (polynomials, trig, exp, log, tanh, sqrt, reciprocal,
interactions) with greedy forward feature selection and ridge regression, plus an
adaptive polynomial-degree fallback and a per-equation time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 320  # safety margin before 330s wall-clock kill


def _elapsed():
    return time.time() - _DEADLINE  # positive when over


def _build_rich_basis(X):
    """Build a rich feature basis: polynomials, trig, exp, log, tanh, sqrt, 1/x, interactions."""
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

    # Trig terms per input
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))

    # Exponential terms (scaled to avoid overflow)
    for j in range(d):
        x_scaled = X[:, j] / 5.0
        cols.append(np.exp(x_scaled))

    # Log terms (inputs are >= 1, so safe)
    for j in range(d):
        cols.append(np.log(X[:, j] + 1e-10))

    # Tanh terms
    for j in range(d):
        cols.append(np.tanh(X[:, j]))

    # Sqrt and reciprocal terms
    for j in range(d):
        cols.append(np.sqrt(X[:, j] + 1e-10))
        cols.append(1.0 / (X[:, j] + 1e-10))

    # Interactions with trig
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * np.sin(X[:, j]))
            cols.append(X[:, i] * np.cos(X[:, j]))

    # Guard against any non-finite values
    M = np.column_stack(cols)
    M = np.where(np.isfinite(M), M, 0.0)
    return M


def _build_poly_basis(X, max_degree=4):
    """Build a pure polynomial feature matrix up to max_degree."""
    d = X.shape[1]
    n = X.shape[0]
    cols = [np.ones(n)]
    for j in range(d):
        cols.append(X[:, j])
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    if max_degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    if max_degree >= 4:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    for l in range(k, d):
                        cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
    M = np.column_stack(cols)
    M = np.where(np.isfinite(M), M, 0.0)
    return M


def _fit_ridge(F, y, alpha):
    """Fit ridge regression; return coefficients and training MSE."""
    k = F.shape[1]
    A = F.T @ F + alpha * np.eye(k)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    pred = F @ w
    mse = np.mean((pred - y) ** 2)
    return w, mse


def _forward_selection(F, y, max_features=20, alpha=1e-6, time_budget=12):
    """Greedy forward feature selection with ridge regression."""
    n_samples, n_features = F.shape
    selected = [0]  # intercept
    best_mse = np.inf
    start_time = time.time()

    while len(selected) < max_features:
        if time.time() - start_time > time_budget:
            break

        best_idx = -1
        best_mse_iter = best_mse
        for idx in range(n_features):
            if idx in selected:
                continue
            trial = selected + [idx]
            F_trial = F[:, trial]
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
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    d = X.shape[1]
    n = X.shape[0]

    # Constant target shortcut
    y_var = np.var(y)
    if y_var < 1e-12:
        c = float(np.mean(y))
        return lambda X_test: np.full(X_test.shape[0], c)

    # --- Primary model: rich basis + forward selection ---
    best_selected = None
    best_w = None
    best_mse = np.inf

    try:
        F_train = _build_rich_basis(X_train)
        selected_idx, w_ridge, mse_fs = _forward_selection(
            F_train, y_train, max_features=20, alpha=1e-6, time_budget=12
        )
        if mse_fs < best_mse:
            best_mse = mse_fs
            best_selected = selected_idx
            best_w = w_ridge
    except Exception:
        pass

    # --- Fallback: adaptive polynomial degree ---
    for degree in [2, 3, 4]:
        if _elapsed() > 10:
            break
        try:
            F_poly = _build_poly_basis(X_train, max_degree=degree)
            w_poly, mse_poly = _fit_ridge(F_poly, y_train, 1e-8)
            if mse_poly < best_mse:
                best_mse = mse_poly
                best_selected = list(range(F_poly.shape[1]))
                best_w = w_poly
                best_is_poly = True
        except Exception:
            continue

    # --- Final safety fallback: degree-2 polynomial least squares ---
    if best_w is None:
        try:
            F_poly = _build_poly_basis(X_train, max_degree=2)
            best_w, *_ = np.linalg.lstsq(F_poly, y, rcond=None)
            best_selected = list(range(F_poly.shape[1]))
        except Exception:
            c = float(np.mean(y))
            return lambda X_test: np.full(X_test.shape[0], c)

    selected_indices = best_selected
    weights = best_w

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        F = _build_rich_basis(X)
        F_sel = F[:, selected_indices]
        out = F_sel @ weights
        # Ensure finite outputs
        out = np.where(np.isfinite(out), out, 0.0)
        return out

    return predict
# EVOLVE-BLOCK-END
