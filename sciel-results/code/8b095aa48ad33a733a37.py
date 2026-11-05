"""Rewritten program using MODEL-AND-OPTIMIZE: fit full degree-5 polynomial with regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Args:
        training_x: flattened array of slow-state values
        training_subgrid_tendency: flattened array of corresponding subgrid tendencies
    
    Returns:
        6 finite real coefficients in ascending powers of x
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit (and 150s kill)
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n == 0:
        return np.zeros(6)
    
    # Try multiple regularization strengths and pick the best
    # Use a simple approach: fit with different lambda values and select based on
    # an internal validation split
    
    # First, normalize x to improve numerical conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for normalized x
    def build_vandermonde(xn, degree=5):
        """Build Vandermonde matrix for polynomial fit in normalized coords."""
        V = np.zeros((len(xn), degree + 1))
        for i in range(degree + 1):
            V[:, i] = xn ** i
        return V
    
    V_norm = build_vandermonde(x_norm, 5)
    
    # K-fold cross-validation for lambda selection
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    
    # Use 5-fold CV if enough data, otherwise fewer folds
    if n >= 30:
        n_folds = 5
    elif n >= 12:
        n_folds = 3
    else:
        n_folds = 2
    
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n
        folds.append(indices[start:end])
    
    # Regularization: penalize higher-order terms more heavily
    reg_weights = np.array([1.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    reg_matrix = np.diag(reg_weights)
    
    # Log-spaced regularization values
    lambdas = [0.0] + list(np.logspace(-12, 4, 25))
    
    best_mse = np.inf
    best_coeffs_norm = None
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 10:
            break
        try:
            mse_sum = 0.0
            count = 0
            valid_fold = True
            for f in range(n_folds):
                val_idx = folds[f]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
                
                V_train = V_norm[train_idx]
                y_train = y[train_idx]
                V_val = V_norm[val_idx]
                y_val = y[val_idx]
                
                if lam == 0.0:
                    coeffs_lam, _, _, _ = np.linalg.lstsq(V_train, y_train, rcond=None)
                else:
                    VtV = V_train.T @ V_train
                    Vty = V_train.T @ y_train
                    A = VtV + lam * reg_matrix
                    try:
                        coeffs_lam = np.linalg.solve(A, Vty)
                    except np.linalg.LinAlgError:
                        coeffs_lam, _, _, _ = np.linalg.lstsq(A, Vty, rcond=None)
                
                if not np.all(np.isfinite(coeffs_lam)):
                    valid_fold = False
                    break
                
                y_val_pred = V_val @ coeffs_lam
                mse_sum += np.mean((y_val_pred - y_val) ** 2)
                count += 1
            
            if not valid_fold or count != n_folds:
                continue
            
            avg_mse = mse_sum / n_folds
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_coeffs_norm = coeffs_lam.copy()
        except:
            continue
    
    if best_coeffs_norm is None:
        # Fallback to OLS on all data
        try:
            coeffs_norm, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
            if np.all(np.isfinite(coeffs_norm)):
                best_coeffs_norm = coeffs_norm
            else:
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    # Convert normalized coefficients to original x-space using stable interpolation
    # Evaluate at Chebyshev points and solve for original coefficients
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_min -= 1.0
        x_max += 1.0
    
    # Chebyshev nodes on [-1, 1] mapped to [x_min, x_max]
    cheb_nodes = np.cos(np.pi * (2 * np.arange(6) + 1) / (2 * 6))
    x_points = 0.5 * (x_max - x_min) * cheb_nodes + 0.5 * (x_max + x_min)
    
    # Evaluate normalized polynomial at these points
    x_points_norm = (x_points - x_mean) / x_std
    V_points = build_vandermonde(x_points_norm, 5)
    y_points = V_points @ best_coeffs_norm
    
    # Solve for original coefficients
    V_orig_points = np.zeros((6, 6))
    for k in range(6):
        V_orig_points[:, k] = x_points ** k
    
    try:
        coeff_orig, _, _, _ = np.linalg.lstsq(V_orig_points, y_points, rcond=None)
    except:
        # Fallback: direct binomial expansion
        coeff_orig = np.zeros(6)
        for i in range(6):
            ci_scaled = best_coeffs_norm[i] / (x_std ** i)
            for k in range(i + 1):
                coeff_orig[k] += ci_scaled * comb(i, k) * ((-x_mean) ** (i - k))
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        try:
            c_lin = np.polyfit(x, y, 1)
            coeff_orig = np.zeros(6)
            coeff_orig[0] = c_lin[1]
            coeff_orig[1] = c_lin[0]
        except:
            coeff_orig = np.zeros(6)
    
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
    
    return coeff_orig
# EVOLVE-BLOCK-END
