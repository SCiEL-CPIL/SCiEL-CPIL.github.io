import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    if n < 6:
        # Not enough data for degree 5, use lower degree
        deg = min(5, n - 1)
        coeff = np.zeros(6)
        if deg >= 0:
            coeff[:deg+1] = np.polynomial.polynomial.polyfit(x, y, deg)
        return coeff
    
    # Use a more stable approach: fit in scaled space, then convert back
    # Scale x to have mean 0 and std 1 for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for degree 5 in scaled x
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Solve least squares
    try:
        coeff_scaled, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        return coeff
    
    # Convert from scaled polynomial to original polynomial
    # y = sum_{j=0}^{5} a_j * ((x - x_mean)/x_std)^j
    # We need: y = sum_{k=0}^{5} c_k * x^k
    
    # First, expand each term: ((x - x_mean)/x_std)^j = (1/x_std^j) * (x - x_mean)^j
    # Then expand (x - x_mean)^j using binomial theorem
    coeff_orig = np.zeros(6)
    
    for j in range(6):
        a_j = coeff_scaled[j] / (x_std ** j)
        # (x - x_mean)^j = sum_{k=0}^{j} C(j,k) * x^k * (-x_mean)^(j-k)
        for k in range(j + 1):
            # Compute binomial coefficient C(j, k)
            binom = 1.0
            for m in range(k):
                binom *= (j - m) / (m + 1)
            coeff_orig[k] += a_j * binom * ((-x_mean) ** (j - k))
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return coeff_orig
    
    # Verify coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback to direct polyfit
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        return coeff
    
    return coeff_orig
