"""Physics-informed sparse basis closure for unresolved Lorenz-96 forcing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import lstsq

def _build_basis(x):
    """Build physics-informed basis matrix: x, x^2, x^3, x^4, x^5 plus interaction terms.
    We use the raw polynomial basis (ascending powers) since the contract requires
    coefficients in ascending powers of x."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    # Standard polynomial basis up to degree 5
    B = np.zeros((n, 6))
    for i in range(6):
        B[:, i] = x ** i
    return B

def _ridge_solve(B, y, lam):
    """Solve ridge regression: min ||Bc - y||^2 + lam*||c||^2"""
    n, p = B.shape
    A = B.T @ B + lam * np.eye(p)
    b = B.T @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c

def _cross_validate_degree_and_reg(x, y, max_degree=5, n_folds=5, 
                                    reg_candidates=None, deadline=None):
    """Select optimal degree and L2 regularization via cross-validation."""
    if reg_candidates is None:
        reg_candidates = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    n = len(x)
    if n < 10:
        return 1, 0.0  # fallback to linear, no reg
    
    # Create folds
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds-1)]
    folds.append(indices[(n_folds-1)*fold_size:])
    
    best_score = np.inf
    best_deg = 1
    best_reg = 0.0
    
    for deg in range(1, max_degree + 1):
        for lam in reg_candidates:
            if deadline is not None and time.time() > deadline - 10:
                break
            fold_errors = []
            for f in range(n_folds):
                val_idx = folds[f]
                tr_idx = np.concatenate([folds[i] for i in range(n_folds) if i != f])
                
                # Build basis for training
                B_tr = np.zeros((len(tr_idx), deg + 1))
                for i in range(deg + 1):
                    B_tr[:, i] = x[tr_idx] ** i
                
                y_tr = y[tr_idx]
                
                # Ridge solve
                A = B_tr.T @ B_tr + lam * np.eye(deg + 1)
                b = B_tr.T @ y_tr
                try:
                    c = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    c = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Evaluate on validation
                B_val = np.zeros((len(val_idx), deg + 1))
                for i in range(deg + 1):
                    B_val[:, i] = x[val_idx] ** i
                pred = B_val @ c
                mse = np.mean((pred - y[val_idx]) ** 2)
                fold_errors.append(mse)
            
            mean_cv = np.mean(fold_errors)
            if mean_cv < best_score:
                best_score = mean_cv
                best_deg = deg
                best_reg = lam
        
        if deadline is not None and time.time() > deadline - 10:
            break
    
    return best_deg, best_reg

def _sparse_l1_selection(x, y, deadline=None):
    """Use L1 regularization (Lasso) to select sparse subset of polynomial terms."""
    n = len(x)
    max_deg = 5
    p = max_deg + 1
    
    # Build full basis
    B = np.zeros((n, p))
    for i in range(p):
        B[:, i] = x ** i
    
    # Try multiple L1 penalties
    l1_candidates = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    best_cv = np.inf
    best_l1 = 0.0
    best_coeffs = np.zeros(p)
    
    # Simple coordinate descent lasso approximation via iterative soft-thresholding
    # For speed, use a simplified approach: try each l1 penalty with a few iterations
    
    for l1 in l1_candidates:
        if deadline is not None and time.time() > deadline - 15:
            break
        
        # Initialize
        c = np.zeros(p)
        
        # A few iterations of coordinate descent
        for _ in range(50):
            if deadline is not None and time.time() > deadline - 15:
                break
            for j in range(p):
                Bj = B[:, j]
                r = y - B @ c + Bj * c[j]
                rho = Bj @ r
                gamma = Bj @ Bj
                if gamma < 1e-12:
                    c[j] = 0.0
                    continue
                # Soft threshold
                if l1 > 0:
                    c[j] = np.sign(rho) * max(abs(rho) - l1, 0.0) / gamma
                else:
                    c[j] = rho / gamma
        
        # Evaluate
        pred = B @ c
        mse = np.mean((pred - y) ** 2)
        if mse < best_cv:
            best_cv = mse
            best_l1 = l1
            best_coeffs = c.copy()
    
    return best_coeffs

def construct(training_x, training_subgrid_tendency):
    """Fit a shared degree-five polynomial approximation to unresolved fast-scale forcing."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Strategy: Cross-validate degree and regularization, then final fit
    # Use a simpler, faster approach given time constraints
    
    # First, try cross-validation with limited candidates
    reg_candidates = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0]
    
    best_cv_score = np.inf
    best_deg = 1
    best_reg = 0.0
    
    n_folds = 5
    if n < 30:
        n_folds = 3
    
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds-1)]
    folds.append(indices[(n_folds-1)*fold_size:])
    
    for deg in range(1, 6):
        if time.time() > deadline - 30:
            break
        for lam in reg_candidates:
            if time.time() > deadline - 30:
                break
            fold_errors = []
            for f in range(n_folds):
                val_idx = folds[f]
                tr_idx = np.concatenate([folds[i] for i in range(n_folds) if i != f])
                
                B_tr = np.zeros((len(tr_idx), deg + 1))
                for i in range(deg + 1):
                    B_tr[:, i] = x[tr_idx] ** i
                
                y_tr = y[tr_idx]
                
                A = B_tr.T @ B_tr + lam * np.eye(deg + 1)
                b = B_tr.T @ y_tr
                try:
                    c = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    c = np.linalg.lstsq(A, b, rcond=None)[0]
                
                B_val = np.zeros((len(val_idx), deg + 1))
                for i in range(deg + 1):
                    B_val[:, i] = x[val_idx] ** i
                pred = B_val @ c
                mse = np.mean((pred - y[val_idx]) ** 2)
                fold_errors.append(mse)
            
            mean_cv = np.mean(fold_errors)
            if mean_cv < best_cv_score:
                best_cv_score = mean_cv
                best_deg = deg
                best_reg = lam
    
    # Final fit with selected degree and regularization
    p = best_deg + 1
    B_full = np.zeros((n, p))
    for i in range(p):
        B_full[:, i] = x ** i
    
    A = B_full.T @ B_full + best_reg * np.eye(p)
    b = B_full.T @ y
    try:
        c_final = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c_final = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Ensure we return 6 coefficients (pad with zeros if degree < 5)
    coeff = np.zeros(6)
    coeff[:p] = c_final
    
    # Final check: ensure all finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
    
    return coeff
# EVOLVE-BLOCK-END
