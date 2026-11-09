# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency.
    
    Strategy:
    1. Use Gaussian kernel smoothing with adaptive bandwidth to denoise the training data.
    2. Use cross-validation to select optimal smoothing parameters.
    3. Fit a degree-5 polynomial to the denoised pseudo-labels.
    4. Return coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin: finish by 100s, limit is 120s
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Normalize x for better numerical conditioning
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_norm = np.zeros_like(x)
        x_center = 0.0
        x_scale = 1.0
    else:
        x_norm = (x - x_min) / x_range
        x_center = x_min
        x_scale = x_range
    
    # Choose evaluation points for kernel smoothing
    # Use more evaluation points for better resolution
    n_eval = min(800, max(200, n // 5))
    x_eval = np.linspace(x_min - 0.2 * x_range, x_max + 0.2 * x_range, n_eval)
    x_eval_norm = (x_eval - x_center) / x_scale if x_scale > 1e-12 else np.zeros_like(x_eval)
    
    # Adaptive bandwidth selection using cross-validation
    # Try multiple bandwidths and select the one that minimizes CV error
    bandwidths = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    def compute_smoothed(x_eval_norm, x_norm, y, bandwidth):
        """Compute kernel-weighted averages at evaluation points."""
        y_smoothed = np.zeros(len(x_eval_norm))
        for i in range(len(x_eval_norm)):
            dist = np.abs(x_eval_norm[i] - x_norm)
            weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
            weight_sum = np.sum(weights)
            if weight_sum > 1e-12:
                y_smoothed[i] = np.sum(weights * y) / weight_sum
            else:
                y_smoothed[i] = np.mean(y)
        return y_smoothed
    
    def cv_error(bandwidth, n_folds=5):
        """Compute cross-validation error for a given bandwidth."""
        if time.time() > deadline - 15:
            return np.inf
        
        # K-fold cross-validation
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        fold_size = n // n_folds
        cv_errors = []
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            if len(train_idx) < 10:
                continue
            
            x_train = x_norm[train_idx]
            y_train = y[train_idx]
            x_val = x_norm[val_idx]
            y_val = y[val_idx]
            
            # Smooth training data
            y_smoothed_train = compute_smoothed(x_val, x_train, y_train, bandwidth)
            
            # Fit polynomial to smoothed training data
            try:
                coeff = np.polynomial.polynomial.polyfit(x_val, y_smoothed_train, 5)
                y_pred = np.polynomial.polynomial.polyval(x_val, coeff)
                mse = np.mean((y_val - y_pred) ** 2)
                cv_errors.append(mse)
            except:
                continue
        
        if len(cv_errors) == 0:
            return np.inf
        return np.mean(cv_errors)
    
    # Select best bandwidth using CV
    best_bandwidth = 0.1
    best_cv_err = np.inf
    
    for bw in bandwidths:
        if time.time() > deadline - 20:
            break
        cv_err = cv_error(bw)
        if cv_err < best_cv_err:
            best_cv_err = cv_err
            best_bandwidth = bw
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback to direct polynomial fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except:
            return np.zeros(6)
    
    # Compute final smoothed data using best bandwidth
    y_smoothed = compute_smoothed(x_eval_norm, x_norm, y, best_bandwidth)
    
    # Fit degree-5 polynomial to the denoised data
    # Use the original x values for the fit
    try:
        coeff = np.polynomial.polynomial.polyfit(x_eval, y_smoothed, 5)
    except:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except:
            coeff = np.zeros(6)
    
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
    
    return coeff
# EVOLVE-BLOCK-END
