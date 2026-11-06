"""Strategy: Ridge-regularized polynomial fit with adaptive bandwidth smoothing and cross-validated lambda selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Determine robust scale
    median_x = np.median(x_norm)
    mad = np.median(np.abs(x_norm - median_x))
    if mad == 0:
        scale = np.std(x_norm)
    else:
        scale = mad * 1.4826
    if scale < 1e-10:
        scale = 1.0
    
    # Construct design matrix for degree 5 polynomial in normalized space
    V = np.vander(x_norm, N=6, increasing=True)
    
    # Time budget check
    if time.time() > deadline - 30:
        # Not enough time, do simple ridge fit
        try:
            A = V.T @ V + 1e-6 * np.eye(6)
            b = V.T @ y
            coeff_norm = np.linalg.solve(A, b)
            if not np.all(np.isfinite(coeff_norm)):
                coeff_norm = np.zeros(6)
            # Transform back to original x space
            return _transform_coeffs(coeff_norm, x_mean, x_std)
        except Exception:
            return np.zeros(6)
    
    # Step 1: Nadaraya-Watson smoothing to reduce noise
    # Choose bandwidth based on data scale
    h_candidates = [scale * n ** -0.2, scale * n ** -0.25, scale * n ** -0.15]
    h_candidates = [max(h, 1e-4) for h in h_candidates]
    
    def nw_smooth(x_input, y_input, bandwidth):
        """Nadaraya-Watson kernel smoother with Gaussian kernel."""
        y_smooth = np.zeros_like(y_input)
        chunk_size = min(500, max(100, 1000000 // len(x_input)))
        for start in range(0, len(x_input), chunk_size):
            end = min(start + chunk_size, len(x_input))
            x_chunk = x_input[start:end]
            diff = x_chunk[:, np.newaxis] - x_input[np.newaxis, :]
            w = np.exp(-0.5 * (diff / bandwidth) ** 2)
            w_sum = np.sum(w, axis=1)
            w_sum[w_sum < 1e-10] = 1e-10
            w_y = np.sum(w * y_input[np.newaxis, :], axis=1)
            y_smooth[start:end] = w_y / w_sum
            if time.time() > deadline - 25:
                break
        return y_smooth
    
    # Try multiple bandwidths and select best via leave-one-out style CV
    # Use a subset for CV to save time
    cv_frac = min(1.0, 20000.0 / n)
    cv_idx = np.random.RandomState(42).choice(n, size=int(n * cv_frac), replace=False)
    x_cv = x_norm[cv_idx]
    y_cv = y[cv_idx]
    
    best_lambda = 1e-4
    best_score = np.inf
    best_coeff = None
    
    # Lambda candidates for ridge regularization
    lambda_candidates = np.logspace(-6, 2, 17)
    
    for h in h_candidates:
        if time.time() > deadline - 15:
            break
        y_smooth = nw_smooth(x_norm, y, h)
        
        for lam in lambda_candidates:
            if time.time() > deadline - 10:
                break
            try:
                # Ridge regression on smoothed targets
                A = V.T @ V + lam * np.eye(6)
                b = V.T @ y_smooth
                coeff_norm = np.linalg.solve(A, b)
                
                if not np.all(np.isfinite(coeff_norm)):
                    continue
                
                # Evaluate on CV subset
                V_cv = np.vander(x_cv, N=6, increasing=True)
                y_pred = V_cv @ coeff_norm
                mse = np.mean((y_cv - y_pred) ** 2)
                
                if mse < best_score:
                    best_score = mse
                    best_lambda = lam
                    best_coeff = coeff_norm.copy()
                    best_h = h
            except Exception:
                continue
    
    # If no valid fit found, fall back to simple ridge
    if best_coeff is None:
        try:
            A = V.T @ V + 1e-4 * np.eye(6)
            b = V.T @ y
            best_coeff = np.linalg.solve(A, b)
            if not np.all(np.isfinite(best_coeff)):
                best_coeff = np.zeros(6)
        except Exception:
            best_coeff = np.zeros(6)
    
    # Transform coefficients back to original x space
    # If y ≈ sum c_i * (x_norm)^i where x_norm = (x - x_mean)/x_std
    # Then y ≈ sum c_i * ((x - x_mean)/x_std)^i
    # = sum c_i / x_std^i * (x - x_mean)^i
    # Expand (x - x_mean)^i in terms of x^j using binomial theorem
    return _transform_coeffs(best_coeff, x_mean, x_std)


def _transform_coeffs(coeff_norm, x_mean, x_std):
    """
    Transform polynomial coefficients from normalized space to original space.
    coeff_norm: coefficients for polynomial in x_norm = (x - x_mean)/x_std
    Returns: coefficients for polynomial in x (original space)
    """
    # We have: f(x) = sum_{i=0}^{5} c_i * ((x - mu)/sigma)^i
    # = sum_{i=0}^{5} c_i / sigma^i * (x - mu)^i
    # Now expand (x - mu)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-mu)^(i-j)
    # So f(x) = sum_{i=0}^{5} sum_{j=0}^{i} c_i / sigma^i * binom(i,j) * x^j * (-mu)^(i-j)
    # = sum_{j=0}^{5} x^j * sum_{i=j}^{5} c_i / sigma^i * binom(i,j) * (-mu)^(i-j)
    
    coeff_orig = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            binom = 1.0
            for k in range(1, i - j + 1):
                binom *= (i - k) / k  # This computes C(i, i-j) = C(i, j)
            # Actually, let's use a cleaner binomial computation
            binom = _binom(i, j)
            s += coeff_norm[i] / (x_std ** i) * binom * ((-x_mean) ** (i - j))
        coeff_orig[j] = s
    
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
    
    return coeff_orig


def _binom(n, k):
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return float(result)
# EVOLVE-BLOCK-END
