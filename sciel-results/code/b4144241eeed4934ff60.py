"""Rewritten program: robust degree-5 polynomial closure with CV-selected regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove any non-finite entries
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Normalize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Normalize y
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Build Vandermonde matrix for degree 5
    V = np.vander(x_norm, N=6, increasing=True)

    # K-Fold Cross-Validation to select lambda
    # We want to minimize validation error, which is a better proxy for generalization
    # than training error.
    
    # Stratified or random split? Random split is fine for large N.
    # We'll use 5-fold CV.
    k_folds = 5
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    best_coeff_norm = None
    best_cv_error = np.inf
    best_lambda = 1e-6
    
    # Log-spaced lambdas
    lambdas = np.logspace(-10, 2, 50)
    
    for lam in lambdas:
        if time.time() > deadline - 10.0:
            break
            
        cv_errors = []
        for i in range(k_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            
            V_train = V[train_idx]
            y_train = y_norm[train_idx]
            V_val = V[val_idx]
            y_val = y_norm[val_idx]
            
            try:
                A = V_train.T @ V_train + lam * np.eye(6)
                b = V_train.T @ y_train
                coeff_norm = np.linalg.solve(A, b)
                
                pred_val = V_val @ coeff_norm
                mse_val = np.mean((pred_val - y_val) ** 2)
                cv_errors.append(mse_val)
            except Exception:
                cv_errors.append(np.inf)
        
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_error:
            best_cv_error = mean_cv_error
            best_lambda = lam
            # Store the coefficients from the last fold? No, we need to refit on all data with best lambda.
            # But we can store the structure. We'll refit at the end.
            
    # Refit on all data with best_lambda
    try:
        A = V.T @ V + best_lambda * np.eye(6)
        b = V.T @ y_norm
        coeff_norm = np.linalg.solve(A, b)
        coeff_orig = _transform_poly_to_original(coeff_norm, x_mean, x_std, y_mean, y_std)
    except Exception:
        # Fallback to OLS if solve fails
        try:
            V_full = np.vander(x, N=6, increasing=True)
            coeff_orig, _, _, _ = np.linalg.lstsq(V_full, y, rcond=None)
        except Exception:
            coeff_orig = np.zeros(6)
            
    # Ensure finite output
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.where(np.isfinite(coeff_orig), coeff_orig, 0.0)

    return coeff_orig


def _transform_poly_to_original(coeff_norm, x_mean, x_std, y_mean, y_std):
    """
    Transform polynomial coefficients from normalized space to original space.
    y = y_mean + y_std * sum_{i=0}^{5} c_i * ((x - x_mean)/x_std)^i
    = y_mean + y_std * sum_{i=0}^{5} c_i * (1/x_std)^i * (x - x_mean)^i
    Expand (x - x_mean)^i using binomial theorem.
    """
    # coeff_norm[i] is the coefficient of (x_norm)^i
    # y = y_mean + y_std * sum_i c_i * (x_norm)^i
    # x_norm = (x - x_mean) / x_std
    # So y = y_mean + y_std * sum_i c_i * (x - x_mean)^i / x_std^i
    # We need to express sum_i c_i * (x - x_mean)^i / x_std^i as sum_j d_j * x^j
    # (x - x_mean)^i = sum_{k=0}^{i} binom(i,k) * x^k * (-x_mean)^(i-k)
    # So the coefficient of x^j in the expansion is:
    # d_j = y_std * sum_{i=j}^{5} c_i / x_std^i * binom(i, j) * (-x_mean)^(i-j)
    d = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            binom_val = _binom(i, j)
            term = coeff_norm[i] / (x_std ** i) * binom_val * ((-x_mean) ** (i - j))
            s += term
        d[j] = y_std * s
    d[0] += y_mean
    return d


def _binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
# EVOLVE-BLOCK-END
