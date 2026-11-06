"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nadaraya_watson_smooth(x, y, bandwidth=None, n_samples=None):
    """Apply Nadaraya-Watson kernel smoothing to denoise y as a function of x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return y.copy()
    
    # Choose bandwidth: ~10% of range, or based on std
    x_range = x.max() - x.min()
    x_std = np.std(x)
    if bandwidth is None:
        bandwidth = max(0.1 * x_range, 0.5 * x_std)
        if bandwidth < 1e-10:
            bandwidth = 1.0
    
    # For large n, use binned/smoothed approach for efficiency
    if n > 2000:
        # Bin-based NW smoothing
        n_bins = min(200, max(50, int(np.sqrt(n))))
        bin_edges = np.linspace(x.min() - bandwidth, x.max() + bandwidth, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_idx = np.digitize(x, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for i in range(n):
            bin_sums[bin_idx[i]] += y[i]
            bin_counts[bin_idx[i]] += 1
        
        bin_means = np.zeros(n_bins)
        mask = bin_counts > 0
        bin_means[mask] = bin_sums[mask] / bin_counts[mask]
        
        # Smooth the binned means with Gaussian kernel
        smooth_y = np.zeros(n)
        for i in range(n):
            weights = np.exp(-0.5 * ((bin_centers - x[i]) / bandwidth) ** 2)
            w_sum = np.sum(weights)
            if w_sum > 1e-10:
                smooth_y[i] = np.sum(weights * bin_means) / w_sum
            else:
                smooth_y[i] = y[i]
        return smooth_y
    else:
        # Direct NW for small n
        smooth_y = np.zeros(n)
        for i in range(n):
            weights = np.exp(-0.5 * ((x - x[i]) / bandwidth) ** 2)
            w_sum = np.sum(weights)
            if w_sum > 1e-10:
                smooth_y[i] = np.sum(weights * y) / w_sum
            else:
                smooth_y[i] = y[i]
        return smooth_y


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 120.0
    
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Scale x for numerical stability
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        sigma = 1.0
    
    x_scaled = (x - mu) / sigma
    
    # Apply Nadaraya-Watson smoothing to denoise the tendency
    t_smoothed = _nadaraya_watson_smooth(x_scaled, t)
    
    # Check deadline
    if time.time() > deadline - 30.0:
        # Fallback to simple linear fit on scaled data
        try:
            c_scaled = np.polynomial.polynomial.polyfit(x_scaled, t_smoothed, 1)
            # Convert back to original scale
            coeff = np.zeros(6)
            # c_scaled[0] + c_scaled[1]*x_scaled where x_scaled = (x-mu)/sigma
            # = c_scaled[0] + c_scaled[1]*(x-mu)/sigma
            # = (c_scaled[0] - c_scaled[1]*mu/sigma) + (c_scaled[1]/sigma)*x
            coeff[0] = c_scaled[0] - c_scaled[1] * mu / sigma
            coeff[1] = c_scaled[1] / sigma
            return coeff
        except:
            return np.zeros(6)
    
    # Fit degree-5 polynomial on scaled x with smoothed t
    try:
        c_scaled = np.polynomial.polynomial.polyfit(x_scaled, t_smoothed, 5)
    except:
        # Fallback to lower degree
        try:
            c_scaled = np.polynomial.polynomial.polyfit(x_scaled, t_smoothed, 3)
            c_scaled = np.pad(c_scaled, (0, 2))
        except:
            try:
                c_scaled = np.polynomial.polynomial.polyfit(x_scaled, t_smoothed, 1)
                c_scaled = np.pad(c_scaled, (0, 4))
            except:
                return np.zeros(6)
    
    # Convert coefficients from scaled polynomial to original x polynomial
    # P(x_scaled) = sum(c_i * x_scaled^i) where x_scaled = (x - mu)/sigma
    # Need to express as sum(d_j * x^j)
    # x_scaled^i = ((x - mu)/sigma)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-mu/sigma)^(i-j)
    
    coeff = np.zeros(6)
    for i in range(6):
        if c_scaled[i] == 0:
            continue
        # Expand ((x - mu)/sigma)^i
        for j in range(i + 1):
            from math import comb
            binom_val = comb(i, j)
            power_term = binom_val * ((-mu) / sigma) ** (i - j)
            coeff[j] += c_scaled[i] * power_term
    
    # Verify finite
    if not np.all(np.isfinite(coeff)):
        return np.zeros(6)
    
    return coeff
# EVOLVE-BLOCK-END
