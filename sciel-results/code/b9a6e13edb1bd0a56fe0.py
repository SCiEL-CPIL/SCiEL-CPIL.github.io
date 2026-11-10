"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def _lasso_path(A, b, lam_grid):
    """
    Compute Lasso solution path for min ||A@c - b||^2 + lam*||c||_1.
    Returns dict mapping lam -> coefficient vector.
    Uses coordinate descent (cycling through coordinates).
    """
    n = A.shape[0]
    p = A.shape[1]
    
    # Precompute column norms squared
    col_norms_sq = np.sum(A**2, axis=0)
    # Precompute A^T @ A diagonal (for coordinate descent)
    AT_A_diag = np.sum(A**2, axis=0)
    AT_b = A.T @ b
    
    solutions = {}
    
    for lam in lam_grid:
        c = np.zeros(p)
        # Coordinate descent
        for it in range(200):
            for j in range(p):
                # Compute residual excluding j
                # r_j = b - A@c + A[:,j]*c[j]
                r_j = AT_b[j] - np.dot(A[:, j], c) + A[j, j] * c[j] if False else (AT_b - A @ c)[j] + A[j, j] * c[j]
                # Actually simpler: maintain residual
                # Let's use a different approach: soft thresholding
                # c_j_new = S( (1/lam) * (b_j - sum_{k!=j} A_jk c_k) ) / (1/lam + A_jj/lam)
                # This is getting complex. Let me use scipy if available.
                pass
        
        # Fallback: use scipy if available, else simple ridge
        solutions[lam] = c
    
    return solutions


def _fit_sparse_physics_basis(x, y):
    """
    Fit using a physics-informed sparse basis with L1 regularization.
    
    Basis functions:
    - x (linear advection)
    - x^3 (cubic nonlinearity)
    - x * mean(x) (mean-field interaction, approximated as x * x_mean)
    - x^2 (quadratic)
    - x^4 (quartic)
    - x^5 (quintic)
    
    We select the optimal subset via L1 regularization.
    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    # Build physics-informed basis in scaled coordinates
    x_scaled = (x - x_mean) / x_std
    
    # Basis functions:
    # 0: constant (intercept)
    # 1: x_scaled (linear)
    # 2: x_scaled^2 (quadratic)
    # 3: x_scaled^3 (cubic - key nonlinear advection)
    # 4: x_scaled * x_mean_scaled (mean-field, but x_mean_scaled is 0, so this is 0)
    #    Actually, let's use x * (x - x_mean) which captures deviation from mean
    #    Or better: use x_scaled^2 - 1 (centered quadratic)
    # Let me redefine the basis more carefully:
    
    # Physics terms:
    # - Linear: x_scaled
    # - Cubic: x_scaled^3 (nonlinear advection)
    # - Mean-field: x_scaled * (x_scaled^2 - 1) = x_scaled^3 - x_scaled (captures deviation)
    # - Quadratic: x_scaled^2
    # - Quartic: x_scaled^4
    # - Quintic: x_scaled^5
    
    # Actually, let's use a simpler physics-motivated set:
    # The unresolved forcing in Lorenz-96 has terms like:
    # - Linear damping: -x
    # - Nonlinear advection: -x^3 (in the fast equation)
    # - Mean-field coupling: x * <Y> which is proportional to x * mean(x)
    
    # Since we're fitting y (tendency) as a function of x (slow state),
    # and the fast dynamics have cubic nonlinearity, let's use:
    # Basis: [1, x_scaled, x_scaled^2, x_scaled^3, x_scaled^4, x_scaled^5]
    # But with L1 regularization to sparsify
    
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Try a range of lambda values for Lasso
    # Use soft thresholding via coordinate descent or scipy
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback to ridge regression
        return _fit_ridge(x, y)
    
    # Lasso via coordinate descent (simple implementation)
    n = len(x)
    p = 6
    
    # Precompute
    AT_A = V.T @ V
    AT_b = V.T @ y
    diag = np.diag(AT_A)
    
    # Grid of lambda values (log-spaced)
    lam_grid = np.logspace(-6, 2, 30)
    
    best_lam = None
    best_cv_score = np.inf
    best_coeff_scaled = None
    
    # Simple cross-validation: split into 5 folds
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // 5
    
    for lam in lam_grid:
        cv_errors = []
        for fold in range(5):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            
            if len(train_idx) < 10:
                continue
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]
            
            # Fit Lasso on training data
            c = np.zeros(p)
            AT_b_train = V_train.T @ y_train
            
            # Coordinate descent
            for it in range(500):
                for j in range(p):
                    # Residual excluding j
                    r_j = AT_b_train[j] - np.dot(V_train[:, j], c) + V_train[j, j] * c[j] if False else 0
                    # Actually: r_j = y_train - V_train@c + V_train[:,j]*c[j]
                    # Let's compute properly
                    residual = y_train - V_train @ c
                    rho_j = np.dot(V_train[:, j], residual) + V_train[j, j] * c[j]
                    
                    # Soft thresholding
                    if lam == 0:
                        c_new = rho_j / diag[j]
                    else:
                        c_new = np.sign(rho_j) * max(abs(rho_j) - lam, 0) / diag[j]
                    
                    c[j] = c_new
            
            # Validate
            y_pred = V_val @ c
            mse = np.mean((y_val - y_pred)**2)
            cv_errors.append(mse)
        
        if len(cv_errors) > 0:
            mean_cv = np.mean(cv_errors)
            if mean_cv < best_cv_score:
                best_cv_score = mean_cv
                best_lam = lam
                # Refit on all data with best lambda
                c = np.zeros(p)
                AT_b_all = V.T @ y
                for it in range(500):
                    for j in range(p):
                        residual = y - V @ c
                        rho_j = np.dot(V[:, j], residual) + V[j, j] * c[j]
                        if lam == 0:
                            c_new = rho_j / diag[j]
                        else:
                            c_new = np.sign(rho_j) * max(abs(rho_j) - lam, 0) / diag[j]
                        c[j] = c_new
                best_coeff_scaled = c
    
    if best_coeff_scaled is None:
        # Fallback: use ridge
        return _fit_ridge(x, y)
    
    # Convert from scaled basis to monomial basis
    coeff_original = np.zeros(6)
    for j in range(6):
        if abs(best_coeff_scaled[j]) < 1e-15:
            continue
        scale_factor = best_coeff_scaled[j] / (x_std ** j)
        for k in range(j + 1):
            coeff_original[k] += scale_factor * comb(j, k) * ((-x_mean) ** (j - k))
    
    return coeff_original


def _fit_ridge(x, y, degree=5, lam=1e-8):
    """Ridge regression fallback."""
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=degree + 1, increasing=True)
    
    reg_weights = np.array([1.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])[:degree + 1]
    R = np.diag(lam * reg_weights)
    
    try:
        A = V.T @ V + R
        b = V.T @ y
        c_scaled = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c_scaled = np.zeros(degree + 1)
    
    coeff_original = np.zeros(degree + 1)
    for j in range(degree + 1):
        if abs(c_scaled[j]) < 1e-15:
            continue
        scale_factor = c_scaled[j] / (x_std ** j)
        for k in range(j + 1):
            coeff_original[k] += scale_factor * comb(j, k) * ((-x_mean) ** (j - k))
    
    return coeff_original


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    training_x = np.asarray(training_x, dtype=np.float64)
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Filter out any non-finite values
    mask = np.isfinite(training_x) & np.isfinite(training_subgrid_tendency)
    x = training_x[mask]
    y = training_subgrid_tendency[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Try sparse physics-informed basis with L1 regularization
    try:
        coeff = _fit_sparse_physics_basis(x, y)
    except Exception:
        # Fallback to ridge regression
        try:
            coeff = _fit_ridge(x, y)
        except Exception:
            # Final fallback: linear fit
            try:
                coeff = np.zeros(6)
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                coeff = np.zeros(6)
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
    
    return coeff
# EVOLVE-BLOCK-END
