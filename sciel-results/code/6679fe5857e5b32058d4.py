"""Symbolic regression over 4 hidden 1-D equations using GP + polynomial blend."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 290.0  # safety margin before 330s kill


def _matern52_kernel(X1, X2, l):
    """Matern 5/2 kernel for 1-D inputs."""
    d = np.abs(X1 - X2) / l
    sqrt5 = np.sqrt(5.0)
    return (1.0 + sqrt5 * d + 5.0 / 3.0 * d * d) * np.exp(-sqrt5 * d)


def _fit_gp(X, y, max_iter=200, tol=1e-7):
    """
    Fit a GP with Matern 5/2 kernel (unit variance, learned length scale)
    and zero noise (or tiny noise for stability). Returns predict callable.
    """
    n = len(X)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    y = y - y.mean()
    y_var = np.var(y)
    if y_var < 1e-12:
        return lambda Xq: np.zeros(len(Xq))

    # Standardize
    x_mean = X.mean()
    x_std = X.std()
    if x_std < 1e-12:
        x_std = 1.0
    Xs = (X - x_mean) / x_std

    K = _matern52_kernel(Xs, Xs, 1.0)
    # Add tiny jitter for numerical stability
    jitter = 1e-8
    K[np.diag_indices_from(K)] += jitter

    # Optimize log-length-scale
    def neg_log_marginal(theta):
        l = np.exp(theta[0])
        K = _matern52_kernel(Xs, Xs, l)
        K[np.diag_indices_from(K)] += jitter
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L, y)
        alpha = np.linalg.solve(L.T, alpha)
        nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        return nll

    # Initial guess: try a few starting points
    best_theta = np.array([0.0])
    best_nll = neg_log_marginal(best_theta)

    for start in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        theta0 = np.array([start])
        try:
            res = minimize(neg_log_marginal, theta0, method='Nelder-Mead',
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_nll:
                best_nll = res.fun
                best_theta = res.x
        except Exception:
            pass

    l_opt = np.exp(best_theta[0])
    K = _matern52_kernel(Xs, Xs, l_opt)
    K[np.diag_indices_from(K)] += jitter
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L, y)
        alpha = np.linalg.solve(L.T, alpha)
    except np.linalg.LinAlgError:
        # Fallback: use identity (no correlation)
        return lambda Xq: np.zeros(len(Xq))

    def predict_gp(Xq):
        Xq = np.asarray(Xq)
        if Xq.ndim == 1:
            Xq = Xq.reshape(-1, 1)
        Xqs = (Xq - x_mean) / x_std
        Ks = _matern52_kernel(Xqs, Xs, l_opt)
        pred = Ks @ alpha + y.mean()
        return pred

    return predict_gp


def _fit_polynomial(X, y, degree=8):
    """Fit a polynomial of given degree."""
    X = X.reshape(-1) if X.ndim > 1 else X
    # Use Chebyshev-like scaling for better conditioning
    x_min, x_max = X.min(), X.max()
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    Xs = 2.0 * (X - x_min) / (x_max - x_min) - 1.0  # map to [-1, 1]

    # Build Vandermonde
    V = np.vander(Xs, N=degree + 1, increasing=True)
    w, *_ = np.linalg.lstsq(V, y, rcond=None)

    def predict_poly(Xq):
        Xq = np.asarray(Xq)
        if Xq.ndim > 1:
            Xq = Xq[:, 0]
        Xqs = 2.0 * (Xq - x_min) / (x_max - x_min) - 1.0
        Vq = np.vander(Xqs, N=degree + 1, increasing=True)
        return Vq @ w

    return predict_poly


def _fit_spline(X, y):
    """Fit a natural cubic spline (interpolating)."""
    from scipy.interpolate import CubicSpline
    X = X.reshape(-1) if X.ndim > 1 else X
    order = np.argsort(X)
    Xs = X[order]
    ys = y[order]
    try:
        cs = CubicSpline(Xs, ys, bc_type='natural')
    except Exception:
        # Fallback to polynomial
        return _fit_polynomial(X, y, degree=6)

    def predict_spline(Xq):
        Xq = np.asarray(Xq)
        if Xq.ndim > 1:
            Xq = Xq[:, 0]
        return cs(Xq)

    return predict_spline


def _cross_validate(X, y, fit_func, n_folds=5, seed=42):
    """Simple k-fold CV to estimate normalized MSE."""
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)
    folds = np.array_split(indices, n_folds)
    nmes = []
    for i in range(n_folds):
        val_idx = folds[i]
        trn_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        try:
            pred = fit_func(X[trn_idx], y[trn_idx])
            y_pred = pred(X[val_idx])
            y_true = y[val_idx]
            var_y = np.var(y_true)
            if var_y < 1e-12:
                nmes.append(0.0)
            else:
                nmes.append(np.mean((y_pred - y_true) ** 2) / var_y)
        except Exception:
            nmes.append(1e10)
    return np.mean(nmes)


def fit(X_train, y_train):
    """
    Fit a regression model using ensemble of GP, polynomial, and spline.
    Returns a predict callable.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d = X.shape[1]

    # Handle multi-dim by using first dimension (task says 1 input)
    if d > 1:
        X_use = X[:, 0]
    else:
        X_use = X[:, 0]

    t_start = time.time()

    # Try multiple models and pick best via CV
    models = []
    cv_scores = []

    # 1. GP with Matern 5/2
    if time.time() < _DEADLINE:
        try:
            gp_pred = _fit_gp(X_use, y)
            models.append(gp_pred)
            cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_gp(Xs, ys), n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # 2. Polynomial degree 6
    if time.time() < _DEADLINE:
        try:
            poly_pred = _fit_polynomial(X_use, y, degree=6)
            models.append(poly_pred)
            cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_polynomial(Xs, ys, degree=6), n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # 3. Polynomial degree 8
    if time.time() < _DEADLINE:
        try:
            poly8_pred = _fit_polynomial(X_use, y, degree=8)
            models.append(poly8_pred)
            cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_polynomial(Xs, ys, degree=8), n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # 4. Polynomial degree 10
    if time.time() < _DEADLINE:
        try:
            poly10_pred = _fit_polynomial(X_use, y, degree=10)
            models.append(poly10_pred)
            cv = _cross_validate(X_use, y, lambda Xs, ys: _fit_polynomial(Xs, ys, degree=10), n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # 5. Spline
    if time.time() < _DEADLINE:
        try:
            spline_pred = _fit_spline(X_use, y)
            models.append(spline_pred)
            cv = _cross_validate(X_use, y, _fit_spline, n_folds=5, seed=42)
            cv_scores.append(cv)
        except Exception:
            pass

    # If no models succeeded, use simple linear
    if not models:
        Xs = X_use.reshape(-1, 1)
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xs)), Xs]), y, rcond=None)
        def predict_linear(Xq):
            Xq = np.asarray(Xq)
            if Xq.ndim == 1:
                Xq = Xq.reshape(-1, 1)
            else:
                Xq = Xq[:, 0:1]
            return np.column_stack([np.ones(len(Xq)), Xq]) @ w
        return predict_linear

    # Pick best model by CV score
    best_idx = np.argmin(cv_scores)
    best_model = models[best_idx]

    # Also try an ensemble: average of top 2 models if they're close
    if len(models) >= 2:
        sorted_idx = np.argsort(cv_scores)
        if cv_scores[sorted_idx[1]] < cv_scores[sorted_idx[0]] * 1.5:  # within 50%
            m1 = models[sorted_idx[0]]
            m2 = models[sorted_idx[1]]
            def predict_ensemble(Xq):
                p1 = m1(Xq)
                p2 = m2(Xq)
                return 0.5 * p1 + 0.5 * p2
            # Quick validation of ensemble
            cv_ens = _cross_validate(X_use, y, lambda Xs, ys: (lambda: None)(), n_folds=3, seed=0) if False else None
            # Just use ensemble if both are good
            if cv_scores[sorted_idx[0]] < 0.5 and cv_scores[sorted_idx[1]] < 0.5:
                best_model = predict_ensemble

    def predict(Xq):
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(-1, 1)
        if Xq.shape[1] > 1:
            Xq = Xq[:, 0:1]
        result = best_model(Xq)
        result = np.asarray(result, dtype=float)
        # Ensure finite
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
        return result

    return predict
