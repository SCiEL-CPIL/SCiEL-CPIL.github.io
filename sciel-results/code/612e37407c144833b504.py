# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial import polynomial as P
from math import comb


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency.
    Uses scaling of the input variable to improve numerical stability
    of the least-squares fit, then transforms coefficients back to
    the original variable space.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle potential NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        return np.zeros(6)
    
    try:
        # Calculate scaling parameters
        mu = np.mean(x)
        sigma = np.std(x)
        
        # Avoid division by zero
        if sigma < 1e-12:
            sigma = 1.0
            
        # Scale x
        x_scaled = (x - mu) / sigma
        
        # Fit polynomial in scaled space
        # P.polyfit returns coefficients in ascending order for the scaled variable z
        c_scaled = P.polyfit(x_scaled, y, 5)
        c_scaled = np.asarray(c_scaled, dtype=np.float64)
        
        if not np.all(np.isfinite(c_scaled)):
            return np.zeros(6)
        
        # Transform coefficients from z-space to x-space
        # We want c_x such that sum(c_x[i] * x**i) = sum(c_z[i] * z**i)
        # z = (x - mu) / sigma = x/sigma - mu/sigma
        # Let a = 1/sigma, b = -mu/sigma
        # z = a*x + b
        # (a*x + b)^k = sum_{j=0}^k binomial(k, j) * (a*x)^j * b^(k-j)
        #             = sum_{j=0}^k binomial(k, j) * a^j * b^(k-j) * x^j
        
        # c_x[j] = sum_{k=j}^5 c_z[k] * binomial(k, j) * a^j * b^(k-j)
        
        a = 1.0 / sigma
        b = -mu / sigma
        
        c_x = np.zeros(6)
        for k in range(6):
            cz = c_scaled[k]
            for j in range(k + 1):
                binom = comb(k, j)
                term = cz * binom * (a ** j) * (b ** (k - j))
                c_x[j] += term
                
        c_x = np.asarray(c_x, dtype=np.float64)
        
        if not np.all(np.isfinite(c_x)):
             # Fallback to unscaled fit if transformation fails
             c = P.polyfit(x, y, 5)
             c = np.asarray(c, dtype=np.float64)
             if np.all(np.isfinite(c)):
                 return c
             return np.zeros(6)
                 
        return c_x
    except Exception:
        return np.zeros(6)
# EVOLVE-BLOCK-END
