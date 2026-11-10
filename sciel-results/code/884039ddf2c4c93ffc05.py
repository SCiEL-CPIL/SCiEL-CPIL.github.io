"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _convert_coeffs_to_original(a, mu, sigma, deg=5):
    """
    Convert polynomial coefficients 'a' for the variable z = (x-mu)/sigma
    to coefficients 'b' for the original variable x.
    y = sum(a_j * z^j) = sum(b_i * x^i)
    """
    b = np.zeros(deg + 1)
    for j in range(deg + 1):
        if a[j] == 0:
            continue
        coeff_a = a[j] / (sigma ** j)
        for k in range(j + 1):
            def _nCr(n, r):
                if r < 0 or r > n:
                    return 0
                if r == 0:
                    return 1
                r = min(r, n - r)
                res = 1
                for i in range(r):
                    res = res * (n - i) // (i + 1)
                return res
            
            binom_val = _nCr(j, k)
            term_coeff = coeff_a * binom_val * ((-mu) ** (j - k))
            b[k] += term_coeff
            
    return b

def _soft_threshold(rho, gamma):
    if rho > gamma:
        return rho - gamma
    elif rho < -gamma:
        return rho + gamma
    else:
        return 0.0

def _solve_lasso(V, y, lam, max_iter=500, tol=1e-6):
    """
    Solve Lasso: min 0.5 * ||y - Vc||^2 + lam * ||c||_1
    V: (N, P), y: (N,), lam: float
    Returns c: (P,)
    """
    N, P = V.shape
    if P == 0:
        return np.array([])
    
    # Precompute Gram matrix G = V.T @ V and V.T @ y
    G = V.T @ V
    Vy = V.T @ y
    diag_G = np.diag(G).copy()
    
    # Avoid division by zero
    diag_G[diag_G < 1e-10] = 1e-10
    
    c = np.zeros(P)
    
    for _ in range(max_iter):
        c_old = c.copy()
        max_change = 0.0
        for j in range(P):
            # r_j = Vy_j - sum_{k!=j} G_jk c_k
            # sum_{k!=j} G_jk c_k = G_j @ c - G_jj c_j
            r_j = Vy[j] - (G[j] @ c - G[j, j] * c[j])
            
            c_new = _soft_threshold(r_j, lam) / diag_G[j]
            
            max_change = max(max_change, abs(c_new - c[j]))
            c[j] = c_new
            
        if max_change < tol:
            break
            
    return c

def _solve_ridge(V, y, lam):
    """
    Solve Ridge: min 0.5 * ||y - Vc||^2 + 0.5 * lam * ||c||_2^2
    V: (N, P), y: (N,), lam: float
    Returns c: (P,)
    """
    N, P = V.shape
    A = V.T @ V + lam * np.eye(P)
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
        if np.all(np.isfinite(c)):
            return c
    except:
        pass
    return np.zeros(P)

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 10:
        return np.array([np.mean(y)] + [0.0]*5)

    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-8:
        sigma = 1.0
    z = (x - mu) / sigma

    deg = 5
    V_full = np.polynomial.polynomial.polyvander(z, deg)
    n = len(z)
    
    # Cross Validation setup
    n_folds = 5
    if n < n_folds * 2:
        n_folds = 2
        if n < 4:
            return np.array([np.mean(y)] + [0.0]*5)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        folds.append((indices[start:end], np.concatenate([indices[:start], indices[end:]])))

    # Try both Lasso and Ridge with CV
    best_cv_mse = np.inf
    best_method = None
    best_lambda = None

    # Ridge lambda search
    ridge_lambdas = np.logspace(-6, 2, 20)
    for lam in ridge_lambdas:
        if time.time() > deadline - 15.0:
            break
            
        cv_mse_sum = 0.0
        valid_folds = 0
        
        for val_idx, train_idx in folds:
            if len(train_idx) < deg + 1:
                continue
                
            V_tr = V_full[train_idx]
            y_tr = y[train_idx]
            V_val = V_full[val_idx]
            y_val = y[val_idx]
            
            try:
                c = _solve_ridge(V_tr, y_tr, lam)
                if not np.all(np.isfinite(c)):
                    continue
                
                pred = V_val @ c
                mse = np.mean((pred - y_val) ** 2)
                cv_mse_sum += mse
                valid_folds += 1
            except Exception:
                continue
        
        if valid_folds > 0:
            avg_mse = cv_mse_sum / valid_folds
            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                best_method = 'ridge'
                best_lambda = lam

    # Lasso lambda search
    lasso_lambdas = np.logspace(-3, 3, 20)
    for lam in lasso_lambdas:
        if time.time() > deadline - 10.0:
            break
            
        cv_mse_sum = 0.0
        valid_folds = 0
        
        for val_idx, train_idx in folds:
            if len(train_idx) < deg + 1:
                continue
                
            V_tr = V_full[train_idx]
            y_tr = y[train_idx]
            V_val = V_full[val_idx]
            y_val = y[val_idx]
            
            try:
                c = _solve_lasso(V_tr, y_tr, lam)
                if not np.all(np.isfinite(c)):
                    continue
                
                pred = V_val @ c
                mse = np.mean((pred - y_val) ** 2)
                cv_mse_sum += mse
                valid_folds += 1
            except Exception:
                continue
        
        if valid_folds > 0:
            avg_mse = cv_mse_sum / valid_folds
            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                best_method = 'lasso'
                best_lambda = lam

    # Refit on all data
    if best_method is not None:
        if best_method == 'ridge':
            try:
                c_std = _solve_ridge(V_full, y, best_lambda)
                if np.all(np.isfinite(c_std)):
                    b_orig = _convert_coeffs_to_original(c_std, mu, sigma, deg)
                    if np.all(np.isfinite(b_orig)):
                        return b_orig
            except Exception:
                pass
        else:  # lasso
            try:
                c_std = _solve_lasso(V_full, y, best_lambda)
                if np.all(np.isfinite(c_std)):
                    b_orig = _convert_coeffs_to_original(c_std, mu, sigma, deg)
                    if np.all(np.isfinite(b_orig)):
                        return b_orig
            except Exception:
                pass

    # Fallback: Ridge with small lambda
    try:
        lam_ridge = 1e-4
        c_std = _solve_ridge(V_full, y, lam_ridge)
        if np.all(np.isfinite(c_std)):
            b_orig = _convert_coeffs_to_original(c_std, mu, sigma, deg)
            if np.all(np.isfinite(b_orig)):
                return b_orig
    except Exception:
        pass

    return np.array([np.mean(y)] + [0.0]*5)

# EVOLVE-BLOCK-END
