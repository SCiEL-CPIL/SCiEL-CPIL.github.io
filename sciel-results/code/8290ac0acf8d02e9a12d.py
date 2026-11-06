import numpy as np
import time
import math

def construct(training_x, training_subgrid_tendency):
    """
    Fits a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Exploits the antisymmetry of the subgrid forcing by fitting an odd polynomial in the
    centered slow state variable.
    
    Parameters:
    training_x: Array of slow-state values.
    training_subgrid_tendency: Array of corresponding subgrid tendency values.
    
    Returns:
    Array of 6 finite real polynomial coefficients in ASCENDING powers of x.
    """
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.zeros(6)

    if x.size < 10:
        # Fallback for very small data: simple linear fit
        try:
            coeff = np.zeros(6)
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff[:2] = c_lin
            if not np.all(np.isfinite(coeff)):
                return np.zeros(6)
            return coeff
        except Exception:
            return np.zeros(6)

    # Standardize x to improve conditioning
    mean_x = np.mean(x)
    std_x = np.std(x)
    if std_x < 1e-12:
        std_x = 1.0
    z = (x - mean_x) / std_x

    # Build design matrix for ODD polynomial in z: z, z^3, z^5
    # This exploits the physical antisymmetry of the subgrid forcing.
    V = np.column_stack((z, z**3, z**5))
    
    n_samples = len(z)
    n_features = 3
    
    # Ridge Regression with CV
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)
    
    n_folds = 5
    fold_size = n_samples // n_folds
    folds = []
    start = 0
    for i in range(n_folds):
        end = start + fold_size if i < n_folds - 1 else n_samples
        folds.append(indices[start:end])
        start = end

    # Lambda grid
    # Estimate scale
    try:
        VtV_full = V.T @ V
        trace_VtV = np.trace(VtV_full)
        if trace_VtV <= 0 or not np.isfinite(trace_VtV):
            trace_VtV = n_samples
    except:
        trace_VtV = n_samples
        
    base_lam = trace_VtV / n_features
    lambdas = np.logspace(-6, 4, num=20) * base_lam
    
    best_lambda = lambdas[0]
    best_cv_error = np.inf
    
    for lam in lambdas:
        if time.time() > deadline - 2.0:
            break
            
        fold_errors = []
        
        for i in range(n_folds):
            val_idx = folds[i]
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]
            
            try:
                A = V_train.T @ V_train + lam * np.eye(n_features)
                b = V_train.T @ y_train
                c_std = np.linalg.solve(A, b)
                
                if not np.all(np.isfinite(c_std)):
                    continue
                
                y_pred = V_val @ c_std
                err = np.mean((y_pred - y_val)**2)
                fold_errors.append(err)
            except np.linalg.LinAlgError:
                fold_errors.append(np.inf)
        
        if len(fold_errors) > 0:
            mean_err = np.mean(fold_errors)
            if mean_err < best_cv_error:
                best_cv_error = mean_err
                best_lambda = lam

    # Final fit
    if time.time() < deadline - 1.0:
        try:
            A_full = V.T @ V + best_lambda * np.eye(n_features)
            b_full = V.T @ y
            c_std = np.linalg.solve(A_full, b_full)
        except np.linalg.LinAlgError:
            c_std = np.linalg.lstsq(V, y, rcond=None)[0]
    else:
        c_std = np.zeros(3)
        try:
            c_std[0] = np.polynomial.polynomial.polyfit(x, y, 1)[1] # Just linear approx
        except:
            pass

    if not np.all(np.isfinite(c_std)):
        return np.zeros(6)

    # Transform coefficients from odd polynomial in z to general polynomial in x
    # P(z) = c0*z + c1*z^3 + c2*z^5
    # z = (x - mu) / sigma
    # We want g(x) = P((x-mu)/sigma)
    
    # Expansion:
    # z = a*x + b with a = 1/sigma, b = -mu/sigma
    # P(z) = c0*z + c1*z^3 + c2*z^5
    
    # Let's use polynomial composition via Vandermonde or manual expansion.
    # Manual expansion is safer for small degree.
    # (a*x + b)^k = sum_{j=0}^k binom(k,j) (a*x)^j b^(k-j)
    
    c_orig = np.zeros(6)
    
    # Helper to add terms from c_std[i] * z^(2*i+1)
    # z^1, z^3, z^5 correspond to indices 0, 1, 2 in c_std
    
    for idx in range(3):
        power = 2 * idx + 1
        ci = c_std[idx]
        if ci == 0: continue
        
        # Expand (a*x + b)^power
        # Term for x^j is: ci * binom(power, j) * a^j * b^(power-j)
        for j in range(power + 1):
            if j < 6: # We only care up to x^5
                binom_val = math.factorial(power) / (math.factorial(j) * math.factorial(power - j))
                a = 1.0 / std_x
                b = -mean_x / std_x
                term_coeff = ci * binom_val * (a ** j) * (b ** (power - j))
                c_orig[j] += term_coeff

    # Ensure finite
    if not np.all(np.isfinite(c_orig)):
        return np.zeros(6)

    return c_orig
