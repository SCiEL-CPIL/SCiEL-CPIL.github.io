import numpy as np
from itertools import product
import time

def _fit_chebyshev(x, y, degree, alpha):
    """Fit polynomial using Chebyshev basis for better conditioning."""
    n = len(x)
    if n < degree + 1:
        degree = max(0, n - 1)
    
    # Map x to [-1, 1] for Chebyshev
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-12:
        x_norm = np.zeros_like(x)
    else:
        x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    
    # Build Chebyshev basis
    T = np.zeros((n, degree + 1))
    T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = x_norm
        for k in range(2, degree + 1):
            T[:, k] = 2.0 * x_norm * T[:, k-1] - T[:, k-2]
    
    if alpha > 0:
        A = T.T @ T
        b = T.T @ y
        reg = alpha * np.eye(degree + 1)
        try:
            c = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            c = np.zeros(degree + 1)
    else:
        try:
            c = np.linalg.lstsq(T, y, rcond=None)[0]
        except Exception:
            c = np.zeros(degree + 1)
    
    return c, x_min, x_max

def _chebyshev_to_monomial(c, x_min, x_max, degree):
    """Convert Chebyshev coefficients to monomial basis."""
    # x_norm = 2*(x - x_min)/(x_max - x_min) - 1
    # Let a = 2/(x_max-x_min), b = -2*x_min/(x_max-x_min) - 1
    # x_norm = a*x + b
    a = 2.0 / (x_max - x_min) if x_max > x_min else 1.0
    b = -2.0 * x_min / (x_max - x_min) - 1.0 if x_max > x_min else -1.0
    
    # T_k(a*x + b) = sum_{j=0}^k d_{k,j} (a*x+b)^j
    # We need to compute the monomial coefficients
    # Use recurrence: T_0=1, T_1=z, T_k=2*z*T_{k-1}-T_{k-2}
    
    # Build polynomial in monomial form
    # Start with T_0 and T_1, build up
    poly = np.zeros(degree + 1)  # coefficients for powers of (a*x+b)
    poly[0] = 1.0  # T_0
    
    if degree >= 1:
        T_prev = np.array([0.0, 1.0])  # T_0 = 1, T_1 = z (coeffs for z^0, z^1)
        T_curr = np.array([0.0, 1.0])
        
        for k in range(2, degree + 1):
            # T_k = 2*z*T_{k-1} - T_{k-2}
            T_next = np.zeros(k + 1)
            # 2*z*T_{k-1}: shift T_{k-1} up by 1 and multiply by 2
            for j in range(k):
                T_next[j+1] += 2.0 * T_curr[j]
            # -T_{k-2}
            for j in range(k-1):
                T_next[j] -= T_prev[j]
            
            T_prev = T_curr
            T_curr = T_next
        
        # T_curr has coefficients for powers of z = a*x+b
        # Now expand (a*x+b)^j to get monomial in x
        result = np.zeros(degree + 1)
        for j in range(degree + 1):
            coeff = c[j] * T_curr[j] if j < len(T_curr) else 0.0
            if coeff == 0:
                continue
            # Expand (a*x+b)^j
            for k in range(j + 1):
                # binom(j,k) * a^k * b^(j-k)
                binom = 1.0
                if k > 0 and k < j:
                    mm = min(k, j-k)
                    num = 1
                    den = 1
                    for t in range(1, mm+1):
                        num *= (j - mm + t)
                        den *= t
                    binom = num / den
                result[k] += coeff * binom * (a ** k) * (b ** (j-k))
        
        return result
    
    return c

def _kfold_cv_mse(x, y, degree, alpha, n_folds=5, seed=42):
    """K-fold cross-validation MSE."""
    n = len(x)
    if n < n_folds * (degree + 2):
        # Not enough data for proper CV, use leave-one-out approximation
        # or just return a large value
        return np.var(y) * 2.0
    
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    
    mse_sum = 0.0
    valid_folds = 0
    
    for k in range(n_folds):
        lo = k * fold_size
        hi = lo + fold_size
        if k == n_folds - 1:
            hi = n
        
        val_idx = indices[lo:hi]
        train_idx = np.concatenate([indices[:lo], indices[hi:]])
        
        if len(train_idx) < degree + 2:
            continue
        
        try:
            c, x_min, x_max = _fit_chebyshev(x[train_idx], y[train_idx], degree, alpha)
            # Predict on validation
            x_val_norm = 2.0 * (x[val_idx] - x_min) / (x_max - x_min) - 1.0 if x_max > x_min else np.zeros_like(x[val_idx])
            T_val = np.zeros((len(val_idx), degree + 1))
            T_val[:, 0] = 1.0
            if degree >= 1:
                T_val[:, 1] = x_val_norm
                for kk in range(2, degree + 1):
                    T_val[:, kk] = 2.0 * x_val_norm * T_val[:, kk-1] - T_val[:, kk-2]
            pred = T_val @ c
            mse_sum += np.mean((pred - y[val_idx]) ** 2)
            valid_folds += 1
        except Exception:
            continue
    
    if valid_folds == 0:
        return np.var(y) * 2.0
    
    return mse_sum / valid_folds

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)

    y_var = np.var(y)
    if y_var < 1e-15:
        y_var = 1e-15

    best_score = np.inf
    best_coeffs = np.zeros(6)

    # Try different degrees and regularization strengths
    degrees = [2, 3, 4, 5]
    alphas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    for deg in degrees:
        if time.time() > deadline:
            break
        for alpha in alphas:
            if time.time() > deadline:
                break
            try:
                cv_mse = _kfold_cv_mse(x, y, deg, alpha)
                score = cv_mse / y_var
                if score < best_score:
                    best_score = score
                    # Fit on full data
                    c, x_min, x_max = _fit_chebyshev(x, y, deg, alpha)
                    coeffs_orig = _chebyshev_to_monomial(c, x_min, x_max, deg)
                    best_coeffs = np.zeros(6)
                    best_coeffs[:deg+1] = coeffs_orig[:min(deg+1, 6)]
            except Exception:
                continue

    # If no valid fit found, fall back to simple linear
    if not np.all(np.isfinite(best_coeffs)) or best_score >= np.inf:
        try:
            # Simple polynomial fit
            c_lin = np.polynomial.polynomial.polyfit(x[:min(n, 2000)], y[:min(n, 2000)], 1)
            best_coeffs = np.zeros(6)
            best_coeffs[:2] = c_lin
        except Exception:
            best_coeffs = np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)

    # Ensure finite
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)

    return best_coeffs.astype(np.float64)
