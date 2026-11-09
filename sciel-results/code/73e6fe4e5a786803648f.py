"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial using kernel-weighted denoising + GPR-inspired approach.
    
    Strategy:
    1. Use Gaussian kernel weighting to estimate E[Y|X] at evaluation points.
    2. Fit a degree-5 polynomial to these denoised pseudo-labels.
    3. Return coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin: finish by 100s, limit is 120s
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 10:
        # Fallback to simple linear fit if too few points
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    # Normalize x for better conditioning
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    x_norm = (x - x_min) / x_range
    
    # Step 1: Kernel-weighted local regression to denoise
    # Choose bandwidth based on data spread
    # Use a quantile-based approach for robust bandwidth selection
    sorted_x = np.sort(x_norm)
    # Use median absolute deviation for bandwidth
    median_x = np.median(x_norm)
    mad = np.median(np.abs(x_norm - median_x))
    if mad < 1e-10:
        mad = 1.0
    # Bandwidth: cover roughly 20% of points on each side
    bandwidth = max(0.05, mad * 1.5)
    
    # Create evaluation grid for denoising
    n_eval = min(2000, n)
    eval_x = np.linspace(0.01, 0.99, n_eval)
    
    # Compute kernel-weighted mean at each eval point
    eval_y = np.zeros(n_eval)
    for i, xi in enumerate(eval_x):
        # Gaussian kernel weights
        diffs = x_norm - xi
        weights = np.exp(-0.5 * (diffs / bandwidth) ** 2)
        w_sum = weights.sum()
        if w_sum > 1e-12:
            eval_y[i] = np.dot(weights, y) / w_sum
        else:
            eval_y[i] = y.mean()
    
    # Step 2: Fit degree-5 polynomial to denoised pseudo-labels
    # Use the original x values mapped to normalized scale, then convert back
    # Fit on normalized scale for numerical stability
    A = np.vander(eval_x, 6, increasing=True)
    # Weighted least squares with uniform weights on pseudo-labels
    try:
        coeff_norm, residuals, rank, sv = np.linalg.lstsq(A, eval_y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to direct polynomial fit
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    # Convert from normalized x to original x scale
    # If x_norm = (x - x_min) / x_range, then x = x_norm * x_range + x_min
    # Polynomial in x_norm: sum(c_i * x_norm^i)
    # We need polynomial in x: sum(d_i * x^i)
    # x_norm = (x - x_min) / x_range
    # So we need to convert coefficients from x_norm basis to x basis
    
    # Use polynomial composition
    # p(x_norm) = sum(c_i * x_norm^i)
    # x_norm = (x - x_min) / x_range = a*x + b where a = 1/x_range, b = -x_min/x_range
    # We want q(x) = p(a*x + b)
    
    a = 1.0 / x_range
    b = -x_min / x_range
    
    # Convert using polynomial composition
    # Start with the constant term and build up
    # q(x) = sum_{i=0}^{5} c_i * (a*x + b)^i
    
    coeff_orig = np.zeros(6)
    for i in range(6):
        if abs(coeff_norm[i]) < 1e-15:
            continue
        # (a*x + b)^i expanded
        # Use binomial theorem
        for j in range(i + 1):
            # C(i,j) * a^j * b^(i-j) * x^j
            if j <= 5:
                from math import comb
                coeff_orig[j] += comb(i, j) * (a ** j) * (b ** (i - j)) * coeff_norm[i]
    
    # Final polish: refine with a high-precision least squares on original data
    # using the denoised values as targets but with original x
    # Actually, let's do a final refinement using the pseudo-labels approach
    # but with the full training data weighted by kernel
    
    # Check time
    if time.time() > deadline - 10:
        return coeff_orig
    
    # Refinement: use the denoised eval points to do a weighted fit on original data
    # Create weights based on how well the current polynomial fits
    pred = np.polynomial.polynomial.polyval(x, coeff_orig)
    residuals = y - pred
    # Use kernel weights to smooth the residual field
    residual_mean = np.zeros(n)
    for i in range(n):
        diffs = x_norm - x_norm[i]
        weights = np.exp(-0.5 * (diffs / bandwidth) ** 2)
        w_sum = weights.sum()
        if w_sum > 1e-12:
            residual_mean[i] = np.dot(weights, residuals) / w_sum
        else:
            residual_mean[i] = 0
    
    # Update y with denoised residuals
    y_refined = pred + residual_mean
    
    # Refit polynomial on refined data
    A_full = np.vander(x_norm, 6, increasing=True)
    try:
        coeff_norm2, _, _, _ = np.linalg.lstsq(A_full, y_refined, rcond=None)
    except np.linalg.LinAlgError:
        return coeff_orig
    
    # Convert back
    coeff_orig2 = np.zeros(6)
    for i in range(6):
        if abs(coeff_norm2[i]) < 1e-15:
            continue
        for j in range(i + 1):
            if j <= 5:
                from math import comb
                coeff_orig2[j] += comb(i, j) * (a ** j) * (b ** (i - j)) * coeff_norm2[i]
    
    # Check time
    if time.time() > deadline - 5:
        return coeff_orig2
    
    # Final high-precision polish: use scipy if available for more stable fit
    try:
        from scipy.optimize import least_squares
        # This might be overkill, so just return the refined result
        pass
    except ImportError:
        pass
    
    return coeff_orig2
# EVOLVE-BLOCK-END
