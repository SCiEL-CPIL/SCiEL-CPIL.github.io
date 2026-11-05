"""Model-and-optimize: Scaled Power Polynomial closure for Lorenz-96 subgrid tendency."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import UnivariateSpline


def construct(training_x, training_subgrid_tendency):
    """
    Fit a degree-5 polynomial using a scaled power basis for numerical stability.
    
    Strategy:
    1. Scale x to z in [-1, 1] using min-max normalization.
    2. Fit a degree-5 polynomial P(z) = sum a_i * z^i using np.polynomial.polynomial.polyfit.
    3. Analytically transform coefficients from z-basis to x-basis using binomial expansion.
    4. This avoids ill-conditioning from large x values and grid-fitting errors.
    """
    t_start = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    n = len(x)
    
    if n < 6:
        # Not enough points for degree 5, fallback to lower degree
        deg = min(5, n - 1)
        if deg < 0:
            return np.zeros(6)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            # Pad to length 6
            full_coeff = np.zeros(6)
            full_coeff[:deg+1] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6)

    # Structural Change: Denoise targets using a cubic spline
    # This reduces the impact of high-frequency noise on the global polynomial fit
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    y_smoothed = y_sorted
    try:
        # Use UnivariateSpline for denoising
        # s is the smoothing factor. Larger s -> more smoothing.
        # Heuristic: s proportional to n
        s_val = max(1.0, n * 0.05)
        spl = UnivariateSpline(x_sorted, y_sorted, s=s_val)
        y_smoothed = spl(x_sorted)
        
        # Ensure finite values
        if not np.all(np.isfinite(y_smoothed)):
            y_smoothed = y_sorted
    except Exception:
        # Fallback to simple moving average if spline fails
        window_size = max(5, n // 10)
        if window_size % 2 == 0:
            window_size += 1
        kernel = np.ones(window_size) / window_size
        y_smoothed = np.convolve(y_sorted, kernel, mode='same')

    # 1. Scaling: Map x to z in [-1, 1]
    x_min = np.min(x_sorted)
    x_max = np.max(x_sorted)
    x_range = x_max - x_min
    
    if x_range < 1e-10:
        # Constant x, fit a constant
        mean_y = np.mean(y_smoothed)
        coeff = np.zeros(6)
        coeff[0] = mean_y
        return coeff
    
    # z = (x - center) / scale
    x_center = (x_min + x_max) / 2.0
    x_scale = x_range / 2.0
    
    z = (x_sorted - x_center) / x_scale
    
    # 2. Fit polynomial in z to the smoothed data
    degree = 5
    try:
        # polyfit returns coefficients in ascending powers: [a_0, a_1, ..., a_5]
        # such that P(z) = a_0 + a_1*z + ... + a_5*z^5
        a = np.polynomial.polynomial.polyfit(z, y_smoothed, degree)
    except Exception:
        # Fallback to lower degree if rank deficient
        try:
            a = np.polynomial.polynomial.polyfit(z, y_smoothed, degree - 1)
            a = np.append(a, 0.0)
        except Exception:
            a = np.zeros(6)
            if n >= 2:
                a[:2] = np.polynomial.polynomial.polyfit(z, y_smoothed, 1)

    # 3. Transform coefficients from z to x
    # We have P(z) = sum_{i=0}^5 a_i * z^i
    # z = (x - c) / s = (1/s) * (x - c)
    # z^i = (1/s^i) * (x - c)^i
    # (x - c)^i = sum_{j=0}^i binom(i, j) * x^j * (-c)^(i-j)
    #
    # So P(x) = sum_{i=0}^5 a_i * (1/s^i) * sum_{j=0}^i binom(i, j) * x^j * (-c)^(i-j)
    # Let b_j be the coefficient of x^j in P(x).
    # b_j = sum_{i=j}^5 a_i * (1/s^i) * binom(i, j) * (-c)^(i-j)
    
    b = np.zeros(6)
    for i in range(6):
        if a[i] == 0:
            continue
        coeff_scale = a[i] / (x_scale ** i)
        for j in range(i + 1):
            # binom(i, j)
            binom_val = 1.0
            # Calculate binom(i, j) efficiently for small i
            if j > i - j:
                j_temp = i - j
            else:
                j_temp = j
            for k in range(1, j_temp + 1):
                binom_val *= (i - j_temp + k) / k
            
            term = coeff_scale * binom_val * ((-x_center) ** (i - j))
            b[j] += term
    
    # Ensure finite
    if not np.all(np.isfinite(b)):
        b = np.zeros(6)
        try:
            b[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
    
    return b
# EVOLVE-BLOCK-END
