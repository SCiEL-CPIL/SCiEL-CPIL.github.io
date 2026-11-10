"""Symbolic regression via sparse RBF + polynomial + Fourier basis with exact least squares."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _kmeans(X, k, max_iter=50, seed=0):
    """Simple k-means clustering."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if k >= n:
        return X.copy()
    # K-means++ initialization
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.randint(n)]
    for j in range(1, k):
        d2 = np.min(cdist(X, centers[:j], 'sqeuclidean'), axis=1)
        d2 = d2 / (d2.sum() + 1e-30)
        centers[j] = X[rng.choice(n, p=d2)]
    for _ in range(max_iter):
        d2 = cdist(X, centers, 'sqeuclidean')
        labels = d2.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                new_centers[j] = X[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers

def _rbf_basis(X, centers, sigma):
    """RBF basis matrix: exp(-||x - c||^2 / (2 sigma^2))."""
    d2 = cdist(X, centers, 'sqeuclidean')
    return np.exp(-d2 / (2.0 * sigma * sigma))

def _poly_basis(X, degree, d):
    """Monomial polynomial basis of total degree up to `degree` in d dims."""
    # Generate all monomials of total degree <= degree
    from itertools import product
    monomials = []
    for total_deg in range(degree + 1):
        # compositions of total_deg into d parts
        for combo in _compositions(total_deg, d):
            monomials.append(combo)
    basis = np.ones((X.shape[0], 1))  # constant term
    for exponents in monomials:
        if sum(exponents) == 0:
            continue
        term = np.ones(X.shape[0])
        for i, e in enumerate(exponents):
            if e > 0:
                term = term * (X[:, i] ** e)
        basis = np.hstack([basis, term[:, None]])
    return basis

def _compositions(n, k):
    """All non-negative integer compositions of n into k parts."""
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest

def _fourier_basis(X, max_freq, d):
    """Sin/cos Fourier basis up to max_freq per dimension."""
    basis = [np.ones((X.shape[0], 1))]
    for dim in range(d):
        for f in range(1, max_freq + 1):
            x = X[:, dim]
            basis.append(np.sin(2.0 * np.pi * f * x)[:, None])
            basis.append(np.cos(2.0 * np.pi * f * x)[:, None])
    return np.hstack(basis)

def _solve_ls(A, b, lam):
    """Solve (A^T A + lam I) w = A^T b with fallback."""
    try:
        AAT = A.T @ A + lam * np.eye(A.shape[1])
        w = np.linalg.solve(AAT, A.T @ b)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(AAT, A.T @ b, rcond=None)
    return w

def _build_basis(X, config):
    """Build combined basis matrix from config."""
    parts = []
    if config.get('use_rbf', True):
        parts.append(_rbf_basis(X, config['centers'], config['sigma']))
    if config.get('use_poly', False):
        parts.append(_poly_basis(X, config['poly_degree'], X.shape[1]))
    if config.get('use_fourier', False):
        parts.append(_fourier_basis(X, config['fourier_max_freq'], X.shape[1]))
    if not parts:
        parts.append(np.ones((X.shape[0], 1)))
    return np.hstack(parts)

def _cv_mse(X, y, config, lam, n_folds=5):
    """K-fold CV MSE for a given basis config and ridge lambda."""
    n = X.shape[0]
    fold_size = n // n_folds
    indices = np.arange(n)
    A_full = _build_basis(X, config)
    cv_mses = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = min((fold + 1) * fold_size, n)
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        A_tr = A_full[train_idx]
        y_tr = y[train_idx]
        w = _solve_ls(A_tr, y_tr, lam)
        pred = A_full[val_idx] @ w
        cv_mses.append(np.mean((pred - y[val_idx]) ** 2))
    return np.mean(cv_mses)

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 320.0  # safety margin before 330s kill
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y = y_train - y_mean

    # Normalize inputs to [0,1]
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min + 1e-12
    X = (X_train - x_min) / x_range

    best_config = None
    best_cv = np.inf
    best_lam = 1e-3

    # Generate candidate centers via k-means with various k
    k_candidates = [5, 8, 12, 16, 20, 30, 40]
    k_candidates = [k for k in k_candidates if k < n]
    if not k_candidates:
        k_candidates = [max(2, n // 2)]

    centers_cache = {}
    for k in k_candidates:
        if time.time() > deadline:
            break
        centers_cache[k] = _kmeans(X, k)

    # Hyperparameter grids
    sigma_grid = [0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0]
    lam_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]

    # Phase 1: RBF-only sweep
    for k in k_candidates:
        if time.time() > deadline:
            break
        centers = centers_cache[k]
        for sigma in sigma_grid:
            if time.time() > deadline:
                break
            for lam in lam_grid:
                if time.time() > deadline:
                    break
                config = {'use_rbf': True, 'centers': centers, 'sigma': sigma,
                          'use_poly': False, 'use_fourier': False}
                cv = _cv_mse(X, y, config, lam)
                if cv < best_cv:
                    best_cv = cv
                    best_config = config.copy()
                    best_config['centers'] = centers.copy()
                    best_lam = lam

    if best_config is None:
        k = k_candidates[0]
        best_config = {'use_rbf': True, 'centers': centers_cache[k], 'sigma': 0.25,
                       'use_poly': False, 'use_fourier': False}
        best_lam = 1e-3

    # Phase 2: Add polynomial basis
    if time.time() < deadline - 40:
        for poly_deg in [2, 3]:
            if time.time() > deadline:
                break
            for sigma in [0.1, 0.25, 0.4]:
                if time.time() > deadline:
                    break
                for lam in [1e-4, 1e-3, 1e-2]:
                    if time.time() > deadline:
                        break
                    config = {'use_rbf': True, 'centers': best_config['centers'],
                              'sigma': sigma, 'use_poly': True, 'poly_degree': poly_deg,
                              'use_fourier': False}
                    cv = _cv_mse(X, y, config, lam)
                    if cv < best_cv:
                        best_cv = cv
                        best_config = config.copy()
                        best_config['centers'] = best_config['centers'].copy()
                        best_lam = lam

    # Phase 3: Add Fourier basis (helps with oscillatory equations)
    if time.time() < deadline - 40:
        for max_freq in [2, 3, 5]:
            if time.time() > deadline:
                break
            for sigma in [0.1, 0.25]:
                if time.time() > deadline:
                    break
                for lam in [1e-4, 1e-3, 1e-2]:
                    if time.time() > deadline:
                        break
                    config = {'use_rbf': True, 'centers': best_config['centers'],
                              'sigma': sigma, 'use_poly': False,
                              'use_fourier': True, 'fourier_max_freq': max_freq}
                    cv = _cv_mse(X, y, config, lam)
                    if cv < best_cv:
                        best_cv = cv
                        best_config = config.copy()
                        best_config['centers'] = best_config['centers'].copy()
                        best_lam = lam

    # Phase 4: Poly-only (no RBF) for purely polynomial equations
    if time.time() < deadline - 30:
        for poly_deg in [2, 3, 4]:
            if time.time() > deadline:
                break
            for lam in [1e-4, 1e-3, 1e-2]:
                if time.time() > deadline:
                    break
                config = {'use_rbf': False, 'use_poly': True, 'poly_degree': poly_deg,
                          'use_fourier': False}
                cv = _cv_mse(X, y, config, lam)
                if cv < best_cv:
                    best_cv = cv
                    best_config = config.copy()
                    best_lam = lam

    # Phase 5: Fourier-only
    if time.time() < deadline - 30:
        for max_freq in [2, 3, 5, 8]:
            if time.time() > deadline:
                break
            for lam in [1e-4, 1e-3, 1e-2]:
                if time.time() > deadline:
                    break
                config = {'use_rbf': False, 'use_poly': False,
                          'use_fourier': True, 'fourier_max_freq': max_freq}
                cv = _cv_mse(X, y, config, lam)
                if cv < best_cv:
                    best_cv = cv
                    best_config = config.copy()
                    best_lam = lam

    # Phase 6: Refine RBF sigma and lambda around best
    if time.time() < deadline - 20:
        base_sigma = best_config.get('sigma', 0.25)
        for sigma_mult in [0.6, 0.8, 1.2, 1.5]:
            if time.time() > deadline:
                break
            for lam_mult in [0.5, 2.0]:
                if time.time() > deadline:
                    break
                new_config = best_config.copy()
                new_config['centers'] = best_config['centers'].copy()
                if new_config.get('use_rbf', True):
                    new_config['sigma'] = base_sigma * sigma_mult
                new_lam = best_lam * lam_mult
                cv = _cv_mse(X, y, new_config, new_lam)
                if cv < best_cv:
                    best_cv = cv
                    best_config = new_config
                    best_lam = new_lam

    # Final fit on all data
    A_final = _build_basis(X, best_config)
    w_final = _solve_ls(A_final, y, best_lam)

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test[None, :]
        Xt = (X_test - x_min) / x_range
        A_test = _build_basis(Xt, best_config)
        return A_test @ w_final + y_mean

    return predict
# EVOLVE-BLOCK-END
