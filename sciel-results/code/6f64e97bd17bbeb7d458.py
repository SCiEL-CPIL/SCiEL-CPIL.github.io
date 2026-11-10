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
        reg[0, 0] += 1e-12
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
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
    b_const = -mu / sigma
    
    for k in range(degree + 1):
        if c_centered[k] == 0:
            continue
        # Expand (a*x + b)^k
        # Coefficients of (a*x + b)^k are binomial coefficients * a^i * b^(k-i) for x^i
        for i in range(k + 1):
            # Calculate binom(k, i)
            if i == 0:
                binom_k_i = 1.0
            elif i == k:
                binom_k_i = 1.0
            else:
                # Use multiplicative formula for numerical stability
                binom_k_i = 1.0
                for j in range(i):
                    binom_k_i *= (k - j) / (j + 1)
            
            term_coeff = binom_k_i * (a ** i) * (b_const ** (k - i))
            coeff_orig[i] += c_centered[k] * term_coeff
            
    return coeff_orig

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
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
    best_score = np.inf
    
    n = len(x)
    
    # Use internal cross-validation to select degree and regularization
    if n > 50:
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        n_folds = 5
        fold_size = n // n_folds
        folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
        
        # Candidate degrees and regularization strengths
        degrees = [5, 4, 3, 2]
        alphas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        
        best_cv_score = np.inf
        best_params = (5, 1e-4)  # Default
        
        for deg in degrees:
            for alpha in alphas:
                if time.time() > deadline:
                    break
                
                cv_errors = []
                for i in range(n_folds):
                    val_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                    
                    x_train = x[train_idx]
                    y_train = y[train_idx]
                    x_val = x[val_idx]
                    y_val = y[val_idx]
                    
                    try:
                        c = _fit_centered(x_train, y_train, deg, alpha)
                        y_pred = np.polynomial.polynomial.polyval(x_val, c)
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
            if len(best_coeff) < 6:
                padded = np.zeros(6)
                padded[:len(best_coeff)] = best_coeff
                best_coeff = padded
        except Exception:
            try:
                best_coeff = np.zeros(6)
                best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except:
                best_coeff = np.zeros(6)
    else:
        # Small sample, use moderate regularization
        try:
            best_coeff = _fit_centered(x, y, 5, 1e-3)
        except Exception:
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
