"""Combined closure: Spline-denoised polynomial fit with adaptive regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import UnivariateSpline

def construct(training_x, training_subgrid_tendency):
    """
    Build a shared degree-5 polynomial approximation to the unresolved
    fast-scale forcing of a two-scale Lorenz-96 model.

    Returns:
        coeff: array of 6 floats, coefficients c_0..c_5 for sum c_i * x**i
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill (Evaluator kills at 150s)

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Ensure unique x for spline fitting if necessary, or just proceed.
    # UnivariateSpline requires x to be strictly increasing.
    # We will sort x and y together for the spline fit, but keep original order for polynomial fit?
    # No, the polynomial fit is global, so order doesn't matter.
    # For spline, we need sorted x.
    
    # Check if x has duplicates. If so, we might need to aggregate or use a different smoother.
    # Let's assume x is continuous enough. If not, we can bin.
    # For robustness, let's bin the data for the smoothing step.
    
    # --- Step 1: Denoise y using a smooth spline ---
    # We try a few smoothing factors 's' for the spline.
    # s=0 is interpolation (no smoothing), s>0 is smoothing.
    # We want to remove high-frequency noise but keep the trend.
    
    # Sort data for spline
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicate x values for spline by averaging y
    unique_x, unique_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) < n:
        # Bin the data
        y_binned = np.zeros(len(unique_x))
        np.add.at(y_binned, unique_indices, y_sorted)
        y_binned /= counts
        x_smooth_input = unique_x
        y_smooth_input = y_binned
    else:
        x_smooth_input = x_sorted
        y_smooth_input = y_sorted
        
    # If there are too few unique points, fallback to raw data
    if len(x_smooth_input) < 10:
        x_smooth_input = x
        y_smooth_input = y

    # Candidate smoothing parameters for the spline
    # s is the sum of squared residuals allowed.
    # Larger s -> smoother curve.
    # We estimate a reasonable range based on noise level.
    y_var = np.var(y_smooth_input)
    # Heuristic: s ranges from 0 to n * y_var
    s_candidates = [0, 10, 100, 1000, 10000, 100 * y_var, 1000 * y_var]
    # Ensure s is non-negative
    s_candidates = [max(0, s) for s in s_candidates]
    # Remove duplicates
    s_candidates = list(set(s_candidates))
    
    # Store smoothed y values for each s
    smoothed_y_dict = {}
    for s in s_candidates:
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            # Fit spline
            # k=3 (cubic) is standard
            spl = UnivariateSpline(x_smooth_input, y_smooth_input, s=s, k=3)
            # Evaluate on original x to get smoothed targets
            y_smoothed = spl(x)
            if np.all(np.isfinite(y_smoothed)):
                smoothed_y_dict[s] = y_smoothed
        except Exception:
            continue
            
    if not smoothed_y_dict:
        # Fallback to raw y if spline failed
        smoothed_y_dict = {0: y}

    # --- Step 2: Fit degree-5 polynomial to smoothed data ---
    # We will perform Ridge regression for each smoothed target and select the best one
    # using a simple validation split.
    
    # Split data for validation (50/50)
    rng = np.random.RandomState(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_size = n // 2
    train_idx = idx[:fold_size]
    val_idx = idx[fold_size:]
    
    x_train = x[train_idx]
    y_train_raw = y[train_idx]
    x_val = x[val_idx]
    y_val_raw = y[val_idx]
    
    # Compute powers for train and val
    powers_train = np.stack([x_train**i for i in range(6)], axis=1)
    powers_val = np.stack([x_val**i for i in range(6)], axis=1)
    
    best_coeff = np.zeros(6)
    best_mse = np.inf
    
    # Regularization lambdas
    # Adaptive lambda based on trace
    trace_term = np.trace(powers_train.T @ powers_train) / len(x_train)
    if trace_term <= 0:
        trace_term = 1.0
        
    lambda_multipliers = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    
    # Evaluate combinations of (smoothing_s, lambda)
    for s, y_smoothed in smoothed_y_dict.items():
        if time.time() - start_time > DEADLINE - 10:
            break
            
        y_train_smooth = y_smoothed[train_idx]
        
        for lm in lambda_multipliers:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            lam = lm * trace_term
            A_reg = powers_train.T @ powers_train + lam * np.eye(6)
            b_vec = powers_train.T @ y_train_smooth
            
            try:
                coeff_candidate = np.linalg.solve(A_reg, b_vec)
            except np.linalg.LinAlgError:
                continue
                
            if not np.all(np.isfinite(coeff_candidate)):
                continue
                
            # Calculate validation MSE using RAW y for validation?
            # The task says "minimize normalized error on independent validation trajectories".
            # The true target is the raw subgrid tendency.
            # So we should evaluate the polynomial's performance against the RAW y_val.
            
            pred_val = powers_val @ coeff_candidate
            mse_val = np.mean((pred_val - y_val_raw)**2)
            
            if mse_val < best_mse:
                best_mse = mse_val
                best_coeff = coeff_candidate

    # If no valid candidate found, fallback to simple Ridge on raw data
    if np.all(best_coeff == 0) or not np.all(np.isfinite(best_coeff)):
        powers_all = np.stack([x**i for i in range(6)], axis=1)
        A_all = powers_all.T @ powers_all
        b_all = powers_all.T @ y
        lam_final = 1e-4 * (np.trace(A_all) / n)
        A_reg_final = A_all + lam_final * np.eye(6)
        try:
            best_coeff = np.linalg.solve(A_reg_final, b_all)
        except np.linalg.LinAlgError:
            best_coeff = np.zeros(6)

    # Final safety check
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)

    return best_coeff.astype(np.float64)
# EVOLVE-BLOCK-END
