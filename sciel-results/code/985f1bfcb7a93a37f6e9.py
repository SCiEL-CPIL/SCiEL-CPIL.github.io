"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    # Convert inputs to 1D numpy arrays of floats
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Ensure arrays are the same length and non-empty
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        # Fallback to zeros if data is invalid
        return np.zeros(6)
        
    # Remove non-finite values to avoid errors in polyfit
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        # Not enough points for a degree 5 fit
        # Fall back to linear or constant
        if len(x) >= 2:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            return coeff
        elif len(x) >= 1:
            coeff = np.zeros(6)
            coeff[0] = np.mean(y)
            return coeff
        else:
            return np.zeros(6)

    try:
        # Center and scale x for numerical stability
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-12:
            x_std = 1.0
        
        x_scaled = (x - x_mean) / x_std
        
        # Fit degree-5 polynomial in scaled space
        coeff_scaled = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
        
        # Convert from scaled basis to original x basis
        # p(x) = sum_{i=0}^{5} c_i * ((x - mu)/sigma)^i
        #      = sum_{i=0}^{5} c_i * (x - mu)^i / sigma^i
        # Expand (x - mu)^i = sum_{k=0}^{i} binom(i,k) * x^k * (-mu)^(i-k)
        
        coeff = np.zeros(6)
        for i in range(6):
            ci = coeff_scaled[i]
            if abs(ci) < 1e-15:
                continue
            inv_sigma_i = 1.0 / (x_std ** i)
            for k in range(i + 1):
                # binom(i, k)
                binom_val = 1.0
                for j in range(k):
                    binom_val *= (i - j) / (j + 1)
                # (-mu)^(i-k)
                if i - k == 0:
                    neg_mu_pow = 1.0
                else:
                    neg_mu_pow = (-x_mean) ** (i - k)
                coeff[k] += ci * binom_val * neg_mu_pow * inv_sigma_i
        
        if not np.all(np.isfinite(coeff)):
            # Fallback to direct fit
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        
        return coeff
    except Exception:
        # Fallback to linear fit if anything goes wrong
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            return coeff
        except:
            return np.zeros(6)
# EVOLVE-BLOCK-END
