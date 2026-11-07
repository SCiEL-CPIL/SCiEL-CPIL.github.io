# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline


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


def _fit_poly(X, y, degree, lam):
    B = _poly_basis(X, degree)
    n, m = B.shape
    A = B.T @ B + lam * np.eye(m)
    w = np.linalg.solve(A, B.T @ y)
    return lambda X: _poly_basis(X, degree) @ w


def _rbf_kernel(X, Y, gamma):
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    return np.exp(-gamma * d2)


def _fit_ridge_rbf(X, y, gamma, lam):
    K = _rbf_kernel(X, X)
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    def predict(Xt):
        return _rbf_kernel(Xt, X, gamma) @ alpha
    return predict


def _fit_ridge_poly_kernel(X, y, degree, lam):
    B = _poly_basis(X, degree)
    K = B @ B.T
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    def predict(Xt):
        Bt = _poly_basis(Xt, degree)
        return (Bt @ B.T) @ alpha
    return predict


def _fit_gp_m52(X, y, max_iter=80):
    """Matern-5/2 GP regression with hyperparameter optimization via LOO-CV."""
    n, d = X.shape
    def build_K(theta):
        l, sigma_f, sigma_n = theta
        l = max(l, 1e-4)
        # pairwise squared distances
        d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        K = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        K[np.diag_indices(n)] += sigma_n ** 2
        return K

    def log_marginal(theta):
        K = build_K(theta)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lmc = 0.5 * np.sum(y * alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        return -lmc

    # initial guess
    scale = np.max(np.ptp(X, axis=0)) + 1e-6
    yvar = np.var(y) + 1e-9
    theta0 = np.array([scale, np.sqrt(yvar), np.sqrt(yvar) * 1e-2])
    res = minimize(log_marginal, theta0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
    l, sigma_f, sigma_n = res.x
    l = max(l, 1e-4)
    K = build_K(res.x)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    def predict(Xt):
        d2 = (np.sum(Xt ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * Xt @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        Ks = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        return Ks @ alpha
    return predict


def _fit_spline_1d(X, y, s):
    """Fit a univariate spline for 1D input."""
    # X is (n, 1)
    x = X[:, 0]
    # Sort by x
    idx = np.argsort(x)
    xs = x[idx]
    ys = y[idx]
    # S = smoothing factor. s=0 is interpolation.
    # We use s = variance * n * factor
    spl = UnivariateSpline(xs, ys, s=s, k=3)
    def predict(Xt):
        return spl(Xt[:, 0])
    return predict


def _cv_mse(fit_func, X, y, n_folds=5, seed=42):
    """
    Calculate Mean Squared Error using k-Fold Cross Validation.
    fit_func should be a callable that takes (X_train_fold, y_train_fold) and returns a predict function.
    """
    n = len(y)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    fold_size = n // n_folds
    last_fold_size = n - fold_size * (n_folds - 1)
    
    mse_sum = 0.0
    count = 0
    start = 0
    
    for i in range(n_folds):
        if i == n_folds - 1:
            end = n
        else:
            end = start + fold_size
            
        val_idx = indices[start:end]
        tr_idx = np.concatenate([indices[:start], indices[end:]])
        
        if len(tr_idx) < 2 or len(val_idx) == 0:
            start = end
            continue
            
        X_tr = X[tr_idx]
        y_tr = y[tr_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        try:
            pred_func = fit_func(X_tr, y_tr)
            preds = pred_func(X_val)
            mse = np.mean((preds - y_val) ** 2)
            if np.isfinite(mse):
                mse_sum += mse
                count += 1
        except Exception:
            pass
            
        start = end
        
    if count == 0:
        return np.inf
    return mse_sum / count


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 300.0
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    yvar = np.var(y) + 1e-12

    candidates = [] # List of (name, fit_func, cv_score)

    # 1) Polynomial Ridge Regression
    # Degrees 2, 3, 4, 5
    # Lams: 1e-6, 1e-4, 1e-2, 1e-1
    for deg in [2, 3, 4, 5]:
        for lam in [1e-6, 1e-4, 1e-2, 1e-1]:
            def fit_poly_func(X_tr, y_tr, d=deg, l=lam):
                return _fit_poly(X_tr, y_tr, d, l)
            name = f"poly_{deg}_l{lam}"
            # Check time
            if time.time() - t0 > DEADLINE - 30:
                break
            try:
                score = _cv_mse(fit_poly_func, X, y)
                candidates.append((name, fit_poly_func, score))
            except Exception:
                pass

    # 2) RBF Kernel Ridge Regression
    # Gammas: 0.1, 0.5, 1, 2, 5
    # Lams: 1e-4, 1e-2, 1e-1
    if time.time() - t0 < DEADLINE - 30:
        for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for lam in [1e-4, 1e-2, 1e-1]:
                def fit_rbf_func(X_tr, y_tr, g=gamma, l=lam):
                    return _fit_ridge_rbf(X_tr, y_tr, g, l)
                name = f"rbf_g{gamma}_l{lam}"
                try:
                    score = _cv_mse(fit_rbf_func, X, y)
                    candidates.append((name, fit_rbf_func, score))
                except Exception:
                    pass

    # 3) Polynomial Kernel Ridge Regression
    # Degrees: 4, 6
    # Lams: 1e-3, 1e-1
    if time.time() - t0 < DEADLINE - 30:
        for deg in [4, 6]:
            for lam in [1e-3, 1e-1]:
                def fit_polyk_func(X_tr, y_tr, d=deg, l=lam):
                    return _fit_ridge_poly_kernel(X_tr, y_tr, d, l)
                name = f"polyK_{deg}_l{lam}"
                try:
                    score = _cv_mse(fit_polyk_func, X, y)
                    candidates.append((name, fit_polyk_func, score))
                except Exception:
                    pass

    # 4) Gaussian Process (Matern-5/2)
    # This is expensive. Only try if we have time and maybe restrict to smaller sets or just one config?
    # Actually, GP fitting involves optimization. CV for GP is very expensive (5 * (opt + solve)).
    # With N=200, one GP fit takes ~0.1-0.5s. 5 folds * 0.5s = 2.5s per config.
    # Let's try 2-3 GP configs.
    if time.time() - t0 < DEADLINE - 60:
        # We need to wrap the GP fit to accept X_tr, y_tr
        def fit_gp_func(X_tr, y_tr, max_it=50):
            return _fit_gp_m52(X_tr, y_tr, max_iter=max_it)
        
        # Try a few variations of max_iter or initial guesses? 
        # The current _fit_gp_m52 uses a fixed initial guess. 
        # Let's just try one GP config to be safe on time.
        name = "gp_m52"
        try:
            score = _cv_mse(fit_gp_func, X, y)
            candidates.append((name, fit_gp_func, score))
        except Exception:
            pass

    # 5) Spline (only for 1D)
    if d == 1 and time.time() - t0 < DEADLINE - 30:
        # Smoothing parameters s
        # s=0 is interpolation (overfit), large s is linear.
        # Try a range.
        y_std = np.std(y)
        for s_factor in [0.01, 0.1, 1.0, 10.0]:
            s_val = s_factor * np.var(y) * n
            def fit_spl_func(X_tr, y_tr, s=s_val):
                return _fit_spline_1d(X_tr, y_tr, s)
            name = f"spl_s{factor}"
            try:
                score = _cv_mse(fit_spl_func, X, y)
                candidates.append((name, fit_spl_func, score))
            except Exception:
                pass

    if not candidates:
        # Fallback
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)

    # Sort candidates by CV score
    candidates.sort(key=lambda x: x[2])
    
    # Select the best one
    best_name, best_fit_func, best_score = candidates[0]
    
    # Refit the best model on ALL training data
    try:
        final_predictor = best_fit_func(X, y)
    except Exception:
        # If refit fails, use the mean
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)

    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        out = final_predictor(Xt)
        out = np.asarray(out, dtype=float).ravel()
        # Ensure finite
        out = np.where(np.isfinite(out), out, np.nanmean(y))
        return out

    return predict
# EVOLVE-BLOCK-END
