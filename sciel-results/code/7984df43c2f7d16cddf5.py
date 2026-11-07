"""Hybrid symbolic regression: polynomial + RBF ensemble with adaptive selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

# Global time tracking
_TIME_START = [0.0]
_TOTAL_DEADLINE = 310.0  # Leave 20s margin before 330s kill

def _remaining():
    return _TOTAL_DEADLINE - (time.time() - _TIME_START[0])

def _poly_basis(X, degree):
    """Return a basis matrix of all monomials up to `degree` (incl. constant)."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for r in range(d):
            for c in range(r, d):
                term = np.ones(n)
                for k in range(r, c + 1):
                    term = term * X[:, k]
                cols.append(term)
    return np.column_stack(cols)

def _rbf_kernel(X, Y, gamma):
    """Compute RBF kernel between X and Y."""
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-gamma * d2)

def _fit_ridge(X, y, lam):
    """Standard linear ridge regression."""
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return lambda Xt: Xt @ w

def _fit_poly_ridge(X, y, degree, lam):
    """Polynomial ridge regression."""
    B = _poly_basis(X, degree)
    n, m = B.shape
    A = B.T @ B + lam * np.eye(m)
    w = np.linalg.solve(A, B.T @ y)
    return lambda Xt: _poly_basis(Xt, degree) @ w

def _fit_rbf_krr(X, y, gamma, lam):
    """Standard RBF kernel ridge regression."""
    K = _rbf_kernel(X, X, gamma)
    n = K.shape[0]
    A = K + lam * np.eye(n)
    alpha = np.linalg.solve(A, y)
    def predict(Xt):
        return _rbf_kernel(Xt, X, gamma) @ alpha
    return predict

def _fit_matern_gp(X, y, nu=2.5, max_iter=30):
    """Matern GP regression with hyperparameter optimization."""
    n, d = X.shape
    
    def build_K(theta):
        l, sigma_f, sigma_n = theta
        l = max(l, 1e-4)
        d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        if nu == 2.5:
            K = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        elif nu == 1.5:
            K = (sigma_f ** 2) * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r / l)
        else:  # nu = 5.0
            K = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5/3 * r ** 2 + 5/9 * r ** 3) * np.exp(-np.sqrt(5) * r / l)
        K[np.diag_indices(n)] += sigma_n ** 2
        return K
    
    def log_marginal(theta):
        try:
            K = build_K(theta)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            lmc = 0.5 * np.sum(y * alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            return lmc
        except np.linalg.LinAlgError:
            return 1e10
    
    scale = np.max(np.ptp(X, axis=0)) + 1e-6
    yvar = np.var(y) + 1e-9
    theta0 = np.array([scale, np.sqrt(yvar), np.sqrt(yvar) * 1e-3])
    
    res = minimize(log_marginal, theta0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5})
    l, sigma_f, sigma_n = res.x
    l = max(l, 1e-4)
    K = build_K(res.x)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    def predict(Xt):
        d2 = (np.sum(Xt ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * Xt @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        if nu == 2.5:
            Ks = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        elif nu == 1.5:
            Ks = (sigma_f ** 2) * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r / l)
        else:
            Ks = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5/3 * r ** 2 + 5/9 * r ** 3) * np.exp(-np.sqrt(5) * r / l)
        return Ks @ alpha
    return predict

def _fit_ensemble(X, y):
    """
    Fit multiple models and return an ensemble predictor that uses
    a simple model selection based on training fit.
    """
    n, d = X.shape
    yvar = np.var(y) + 1e-12
    
    candidates = []
    
    # 1) Polynomial models
    for deg in (2, 3, 4):
        if _remaining() < 20:
            break
        for lam in (1e-6, 1e-4, 1e-2, 0.1):
            try:
                p = _fit_poly_ridge(X, y, deg, lam)
                pred = p(X)
                score = np.mean((pred - y) ** 2) / yvar
                candidates.append((score, p))
            except Exception:
                pass
    
    # 2) RBF KRR with various gammas
    if _remaining() > 30:
        for gamma in (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
            for lam in (1e-4, 1e-3, 1e-2, 0.1):
                try:
                    p = _fit_rbf_krr(X, y, gamma, lam)
                    pred = p(X)
                    score = np.mean((pred - y) ** 2) / yvar
                    candidates.append((score, p))
                except Exception:
                    pass
    
    # 3) Matern GP
    if _remaining() > 40:
        for nu in (1.5, 2.5):
            try:
                p = _fit_matern_gp(X, y, nu=nu, max_iter=25)
                pred = p(X)
                score = np.mean((pred - y) ** 2) / yvar
                candidates.append((score, p))
            except Exception:
                pass
    
    if not candidates:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    # Sort by training score
    candidates.sort(key=lambda x: x[0])
    
    # Take top K models and average them for robustness
    top_k = min(3, len(candidates))
    top_models = [p for _, p in candidates[:top_k]]
    
    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        preds = []
        for p in top_models:
            try:
                pred = p(Xt)
                pred = np.asarray(pred, dtype=float).ravel()
                if np.all(np.isfinite(pred)):
                    preds.append(pred)
            except Exception:
                continue
        if not preds:
            return np.full(len(Xt), float(np.mean(y)))
        return np.mean(preds, axis=0)
    
    return predict

def fit(X_train, y_train):
    _TIME_START[0] = time.time()
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    try:
        predict = _fit_ensemble(X, y)
    except Exception:
        c = float(np.mean(y))
        predict = lambda Xt: np.full(len(Xt), c)
    
    def safe_predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        out = predict(Xt)
        out = np.asarray(out, dtype=float).ravel()
        out = np.where(np.isfinite(out), out, np.nanmean(y))
        return out
    
    return safe_predict
# EVOLVE-BLOCK-END
