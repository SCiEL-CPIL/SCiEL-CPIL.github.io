"""Symbolic regression with adaptive Gaussian-RBF basis and cross-validation model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')


def _safe_predict(predict_fn, X):
    """Ensure predictions are finite."""
    X = np.asarray(X, dtype=float)
    try:
        pred = predict_fn(X)
    except Exception:
        return np.zeros(len(np.asarray(X).reshape(-1)))
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return pred


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 330s limit

    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)

    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_vals = X[:, 0].copy()
    y_vals = y.copy()

    # Constant check
    if np.std(y_vals) < 1e-12:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    y_var = np.var(y_vals)
    if y_var < 1e-15:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    x_center = (x_min + x_max) / 2.0

    # Build an adaptive basis:
    #  - Polynomials (up to degree 8)
    #  - Cosine/sine with frequencies adapted to range
    #  - Exponentials
    #  - Gaussian RBFs centered at quantiles of x
    #  - Rational / sqrt / abs / tanh
    basis_names = []
    basis_funcs = []

    # Constant
    basis_names.append('1')
    basis_funcs.append(lambda x: np.ones_like(x, dtype=float))

    # Polynomials up to degree 8
    for d in range(1, 9):
        basis_names.append(f'x^{d}')
        basis_funcs.append(lambda x, d=d: x ** d)

    # Centered polynomials (better conditioned)
    for d in range(1, 9):
        basis_names.append(f'(x-c)^{d}')
        basis_funcs.append(lambda x, d=d: (x - x_center) ** d)

    # Trigonometric with several frequencies
    freqs = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    for f in freqs:
        basis_names.append(f'cos({f}*x)')
        basis_funcs.append(lambda x, f=f: np.cos(f * x))
        basis_names.append(f'sin({f}*x)')
        basis_funcs.append(lambda x, f=f: np.sin(f * x))

    # Exponentials
    for b in [-3.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]:
        basis_names.append(f'exp({b}*x)')
        basis_funcs.append(lambda x, b=b: np.exp(b * x))

    # Gaussian RBFs centered at quantiles
    n_rbf = 16
    centers = np.linspace(x_min, x_max, n_rbf + 2)[1:-1]  # 16 centers inside
    for c in centers:
        basis_names.append(f'gauss_{c:.3f}')
        basis_funcs.append(lambda x, c=c: np.exp(-((x - c) ** 2) / (2.0 * (0.25 * x_range) ** 2)))

    # Additional narrower RBFs
    for c in centers[::2]:
        basis_names.append(f'gauss_n_{c:.3f}')
        basis_funcs.append(lambda x, c=c: np.exp(-((x - c) ** 2) / (2.0 * (0.1 * x_range) ** 2)))

    # Rational
    basis_names.append('1/(x+1)')
    basis_funcs.append(lambda x: 1.0 / (x + 1.0))
    basis_names.append('1/(x-1)')
    basis_funcs.append(lambda x: 1.0 / (x - 1.0))
    basis_names.append('1/(x^2+1)')
    basis_funcs.append(lambda x: 1.0 / (x ** 2 + 1.0))
    basis_names.append('1/(x^2+0.1)')
    basis_funcs.append(lambda x: 1.0 / (x ** 2 + 0.1))

    # Sqrt / abs
    basis_names.append('|x|')
    basis_funcs.append(lambda x: np.abs(x))
    basis_names.append('x*|x|')
    basis_funcs.append(lambda x: x * np.abs(x))
    basis_names.append('sqrt(x^2+1)')
    basis_funcs.append(lambda x: np.sqrt(x ** 2 + 1.0))
    if x_min > -0.5:
        basis_names.append('sqrt(|x|+0.1)')
        basis_funcs.append(lambda x: np.sqrt(np.abs(x) + 0.1))

    # Tanh / sigmoid
    for f in [0.25, 0.5, 1.0, 2.0, 4.0]:
        basis_names.append(f'tanh({f}*x)')
        basis_funcs.append(lambda x, f=f: np.tanh(f * x))
        basis_names.append(f'sigmoid({f}*x)')
        basis_funcs.append(lambda x, f=f: 1.0 / (1.0 + np.exp(-f * x)))

    # Log
    basis_names.append('log(x^2+1)')
    basis_funcs.append(lambda x: np.log(x ** 2 + 1.0))
    if x_min > -0.5:
        basis_names.append('log(|x|+0.1)')
        basis_funcs.append(lambda x: np.log(np.abs(x) + 0.1))

    # Evaluate all basis functions on training data
    n_basis = len(basis_funcs)
    B = np.zeros((n, n_basis))
    for j in range(n_basis):
        try:
            col = basis_funcs[j](x_vals)
            col = np.asarray(col, dtype=float)
            if not np.all(np.isfinite(col)):
                col = np.where(np.isfinite(col), col, 0.0)
            B[:, j] = col
        except Exception:
            B[:, j] = 0.0

    # Drop degenerate columns
    col_var = np.var(B, axis=0)
    valid = col_var > 1e-12
    B = B[:, valid]
    basis_funcs = [basis_funcs[i] for i in range(n_basis) if valid[i]]
    n_valid = len(basis_funcs)

    if n_valid == 0:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    # Cross-validation setup: 5-fold
    rng = np.random.RandomState(42)
    idx_all = np.arange(n)
    rng.shuffle(idx_all)
    folds = np.array_split(idx_all, 5)

    def cv_score(indices, ridge):
        """Return CV normalized MSE for given basis indices and ridge penalty."""
        if time.time() > deadline - 2:
            return np.inf
        try:
            se = 0.0
            for tr, va in zip(range(5), folds):
                train_idx = np.concatenate([folds[i] for i in range(5) if i != tr])
                val_idx = folds[tr]
                A = B[train_idx][:, indices]
                b = y_vals[train_idx]
                # Regularized least squares: (A^T A + ridge I) w = A^T b
                try:
                    w = np.linalg.solve(A.T @ A + ridge * np.eye(len(indices)), A.T @ b)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(A, b, rcond=None)[0]
                pred = B[val_idx][:, indices] @ w
                if not np.all(np.isfinite(pred)):
                    return np.inf
                se += np.sum((pred - y_vals[val_idx]) ** 2)
            return se / (n * y_var)
        except Exception:
            return np.inf

    def train_coefficients(indices, ridge):
        A = B[:, indices]
        try:
            w = np.linalg.solve(A.T @ A + ridge * np.eye(len(indices)), A.T @ y_vals)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, y_vals, rcond=None)[0]
        return w

    # Candidate families:
    # 1. All basis functions with various ridge values
    # 2. Subsets: polynomials only, trig only, RBF only, mixed
    # 3. Random subsets of various sizes

    candidates = []  # (cv_score, indices, ridge)

    ridge_vals = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    # Full basis with ridge
    for r in ridge_vals:
        s = cv_score(np.arange(n_valid), r)
        if np.isfinite(s):
            candidates.append((s, list(range(n_valid)), r))

    # Polynomial-only subsets
    poly_idx = [i for i, nm in enumerate(basis_names) if nm.startswith('x^') or nm.startswith('(x-c)^')]
    if poly_idx:
        for r in [1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
            s = cv_score(np.array(poly_idx), r)
            if np.isfinite(s):
                candidates.append((s, list(poly_idx), r))

    # Trig-only
    trig_idx = [i for i, nm in enumerate(basis_names) if nm.startswith('cos') or nm.startswith('sin')]
    if trig_idx:
        for r in [1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
            s = cv_score(np.array(trig_idx), r)
            if np.isfinite(s):
                candidates.append((s, list(trig_idx), r))

    # RBF-only
    rbf_idx = [i for i, nm in enumerate(basis_names) if nm.startswith('gauss')]
    if rbf_idx:
        for r in [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]:
            s = cv_score(np.array(rbf_idx), r)
            if np.isfinite(s):
                candidates.append((s, list(rbf_idx), r))

    # Random subsets of various sizes
    np.random.seed(123)
    for size in [3, 5, 8, 12, 16, 20]:
        if n_valid < size:
            continue
        n_trials = 15
        for _ in range(n_trials):
            if time.time() > deadline - 5:
                break
            idx = np.random.choice(n_valid, size=size, replace=False)
            for r in [1e-6, 1e-4, 1e-2, 1.0]:
                s = cv_score(np.array(idx), r)
                if np.isfinite(s):
                    candidates.append((s, list(idx), r))

    # If no valid candidates, fall back to constant
    if not candidates:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    # Sort by CV score
    candidates.sort(key=lambda c: c[0])
    best_cv, best_idx, best_ridge = candidates[0]
    best_idx = np.array(best_idx)

    # Train final coefficients on all data
    w = train_coefficients(best_idx, best_ridge)
    if not np.all(np.isfinite(w)):
        # Fallback to least squares
        A = B[:, best_idx]
        w = np.linalg.lstsq(A, y_vals, rcond=None)[0]

    # Also keep a polynomial fallback in case prediction goes wrong
    poly_fallback = None
    for deg in [2, 3, 4, 5]:
        if deg >= n:
            break
        try:
            pc = np.polyfit(x_vals, y_vals, deg)
            pp = np.polyval(pc, x_vals)
            if np.all(np.isfinite(pp)):
                poly_fallback = pc
                break
        except Exception:
            continue

    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
        n_pred = len(X_pred)
        try:
            A = np.ones((n_pred, len(best_idx)))
            for k, idx in enumerate(best_idx):
                try:
                    col = basis_funcs[idx](X_pred)
                    col = np.asarray(col, dtype=float)
                    if not np.all(np.isfinite(col)):
                        col = np.where(np.isfinite(col), col, 0.0)
                    A[:, k + 1] = col
                except Exception:
                    A[:, k + 1] = 0.0
            pred = A @ w
            if not np.all(np.isfinite(pred)):
                pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            if poly_fallback is not None:
                try:
                    return np.polyval(poly_fallback, X_pred)
                except Exception:
                    pass
            return np.zeros(n_pred)

    return predict
# EVOLVE-BLOCK-END
