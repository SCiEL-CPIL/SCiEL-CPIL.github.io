"""Program using GPR denoising followed by degree-5 polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import gaussian_kde


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved subgrid tendency.
    
    Strategy:
    1. Use Gaussian Process Regression with an RBF kernel to estimate the
       conditional expectation E[Y|X] at a set of well-distributed points.
       This denoises the relationship.
    2. Fit a degree-5 polynomial to these pseudo-labels.
    
    Returns:
        coeff: array of 6 coefficients in ascending powers of x.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # If very few samples, just do a direct polynomial fit
    if n < 20:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return np.asarray(coeff, dtype=np.float64)
        except Exception:
            return np.zeros(6)
    
    # Estimate noise level from local residuals
    # Sort x and compute local variance
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    
    # Estimate noise variance using a simple approach: 
    # compute residuals from a local linear fit
    # Use a moving window
    window = max(5, n // 20)
    noise_var_estimates = []
    for i in range(0, n - window, max(1, window // 4)):
        xw = x_sorted[i:i+window]
        yw = y_sorted[i:i+window]
        if len(np.unique(xw)) >= 2:
            # Local linear fit
            p1 = np.polyfit(xw, yw, 1)
            resid = yw - np.polyval(p1, xw)
            noise_var_estimates.append(np.var(resid))
    
    if noise_var_estimates:
        noise_var = np.median(noise_var_estimates)
    else:
        noise_var = np.var(y) * 0.1
    
    # Ensure noise_var is positive
    noise_var = max(noise_var, 1e-10)
    
    # Create evaluation points for GPR: well-distributed points in x range
    x_min, x_max = np.min(x), np.max(x)
    n_eval = min(200, n)
    # Use quantiles to ensure good coverage
    eval_points = np.quantile(x, np.linspace(0, 1, n_eval))
    # Add some random points for robustness
    extra_points = np.random.uniform(x_min, x_max, size=min(50, n))
    eval_points = np.concatenate([eval_points, extra_points])
    eval_points = np.unique(eval_points)
    
    # GPR with RBF kernel
    # Use a simple iterative approach to estimate length scale
    # Start with a reasonable length scale estimate
    # Based on typical spacing
    dx_typical = np.median(np.diff(np.sort(x)))
    if dx_typical <= 0:
        dx_typical = (x_max - x_min) / max(n - 1, 1)
    
    # Try a few length scales and pick the one with best leave-one-out-ish criterion
    # For efficiency, we'll use a fixed set of length scales
    length_scales = np.array([0.5, 1.0, 2.0, 5.0, 10.0]) * dx_typical
    length_scales = length_scales[length_scales > 0]
    if len(length_scales) == 0:
        length_scales = np.array([1.0])
    
    best_ls = length_scales[0]
    best_score = np.inf
    
    # For each length scale, compute the GPR predictions at eval points
    # and measure how well they smooth the data
    for ls in length_scales:
        try:
            # Build kernel matrix
            # K(x, x') = sigma_f^2 * exp(-||x-x'||^2 / (2*ls^2))
            # We need to estimate sigma_f^2 as well. Use marginal likelihood approach
            # simplified: just compute predictions and check consistency
            
            # Compute kernel matrix for training data
            diff = x[:, None] - x[None, :]
            K = np.exp(-diff**2 / (2 * ls**2))
            
            # Estimate signal variance as variance of y
            sigma_f2 = np.var(y)
            if sigma_f2 <= 0:
                sigma_f2 = 1.0
            
            # Add noise
            K_noisy = K + (noise_var / sigma_f2) * np.eye(n)
            
            # Solve for weights
            try:
                alpha = np.linalg.solve(K_noisy, y)
            except np.linalg.LinAlgError:
                continue
            
            # Predict at eval points
            K_star = np.exp(-(eval_points[:, None] - x[None, :])**2 / (2 * ls**2))
            y_pred = K_star @ alpha
            
            # Score: how much variance is explained vs. raw variance
            # We want smooth predictions that capture the trend
            # Use a simple score: correlation between y_pred and local means
            # or just check that y_pred is smooth (low derivative)
            
            # Compute smoothness: sum of squared second differences
            if len(y_pred) > 2:
                second_diff = np.diff(y_pred, 2)
                smoothness = np.sum(second_diff**2) / (len(y_pred) - 2)
            else:
                smoothness = 0
            
            # Also check that predictions don't deviate too much from data
            # Residual at training points
            y_train_pred = np.exp(-(x[:, None] - x[None, :])**2 / (2 * ls**2)) @ alpha
            train_mse = np.mean((y_train_pred - y)**2)
            
            # Combined score: want low train_mse and low smoothness penalty
            # Normalize
            score = train_mse + 0.01 * smoothness
            
            if score < best_score:
                best_score = score
                best_ls = ls
        except Exception:
            continue
    
    # Final GPR prediction with best length scale
    ls = best_ls
    diff = x[:, None] - x[None, :]
    K = np.exp(-diff**2 / (2 * ls**2))
    sigma_f2 = np.var(y)
    if sigma_f2 <= 0:
        sigma_f2 = 1.0
    K_noisy = K + (noise_var / sigma_f2) * np.eye(n)
    
    try:
        alpha = np.linalg.solve(K_noisy, y)
    except np.linalg.LinAlgError:
        # Fallback to direct polynomial fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return np.asarray(coeff, dtype=np.float64)
        except Exception:
            return np.zeros(6)
    
    # Predict at eval points
    K_star = np.exp(-(eval_points[:, None] - x[None, :])**2 / (2 * ls**2))
    y_pseudo = K_star @ alpha
    
    # Fit degree-5 polynomial to pseudo-labels
    try:
        coeff = np.polynomial.polynomial.polyfit(eval_points, y_pseudo, 5)
        coeff = np.asarray(coeff, dtype=np.float64)
        # Ensure finite values
        if not np.all(np.isfinite(coeff)):
            raise ValueError
        return coeff
    except Exception:
        # Fallback to direct polynomial fit on original data
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            coeff = np.asarray(coeff, dtype=np.float64)
            if not np.all(np.isfinite(coeff)):
                raise ValueError
            return coeff
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
