"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fit_centered(x, y, degree, alpha):
    """
    Fit a degree-`degree` polynomial in centered/scaled x, with ridge regularization.
    Returns coefficients in the original x basis (ascending powers).
    """
    n = len(x)
    if n == 0:
        return np.zeros(degree + 1)
    
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    
    xc = (x - mu) / sigma
    
    # Build Vandermonde matrix for centered/scaled variable
    V = np.column_stack([xc**k for k in range(degree + 1)])
    
    A = V.T @ V
    b = V.T @ y
    
    if alpha > 0:
        reg = alpha * np.eye(A.shape[0])
        # Small diagonal for numerical safety
        reg[0, 0] += 1e-12
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if solve fails
            c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)
    else:
        try:
            c_centered = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Convert from centered/scaled basis to original basis
    # p(x) = sum_{k=0}^{degree} c_k * ((x-mu)/sigma)^k
    # Let z = (x-mu)/sigma => x = sigma*z + mu
    # p(x) = sum c_k * z^k
    # We need to express this in powers of x.
    
    coeff_orig = np.zeros(degree + 1)
    a = 1.0 / sigma
    b_val = -mu / sigma
    
    from math import comb
    for k in range(degree + 1):
        if c_centered[k] == 0:
            continue
        # Expand (a*x + b_val)^k
        # Coefficients of (a*x + b_val)^k are binom(k, i) * a^i * b_val^(k-i) for x^i
        for i in range(k + 1):
            binom_k_i = comb(k, i)
            term_coeff = binom_k_i * (a ** i) * (b_val ** (k - i))
            coeff_orig[i] += c_centered[k] * term_coeff
            
    return coeff_orig

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return np.zeros(6)
        
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        if len(x) >= 2:
            coeff = np.zeros(6)
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except:
                coeff[0] = np.mean(y)
            return coeff
        elif len(x) >= 1:
            coeff = np.zeros(6)
            coeff[0] = np.mean(y)
            return coeff
        else:
            return np.zeros(6)

    best_coeff = np.zeros(6)
    n = len(x)
    
    # Do 5-fold CV to select alpha and degree
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n // 5
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(5)]
    
    candidates = []
    for deg in [5, 4, 3, 2]:
        for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            candidates.append((deg, alpha))
    
    best_cv_score = np.inf
    best_params = (5, 1e-3) # Default
    
    for deg, alpha in candidates:
        cv_errors = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]
            
            # Fit
            try:
                c_centered = _fit_centered(x_train, y_train, deg, alpha)
                # Evaluate on validation
                y_pred = np.polynomial.polynomial.polyval(x_val, c_centered)
                mse = np.mean((y_val - y_pred)**2)
                cv_errors.append(mse)
            except Exception:
                cv_errors.append(np.inf)
        
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_score:
            best_cv_score = mean_cv_error
            best_params = (deg, alpha)
            
        if time.time() > deadline:
            break
    
    deg, alpha = best_params
    try:
        best_coeff = _fit_centered(x, y, deg, alpha)
        # Pad to length 6 if deg < 5
        if len(best_coeff) < 6:
            padded = np.zeros(6)
            padded[:len(best_coeff)] = best_coeff
            best_coeff = padded
    except Exception:
        # Fallback
        try:
            best_coeff = np.zeros(6)
            best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except:
            best_coeff = np.zeros(6)
            
    # Ensure finite
    if not np.all(np.isfinite(best_coeff)):
        try:
            best_coeff = np.zeros(6)
            best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except:
            best_coeff = np.zeros(6)
            
    return best_coeff[:6]
# EVOLVE-BLOCK-END
