import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 145.0  # Safety margin from 150s
    
    # Convert to numpy arrays
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Handle length mismatch
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
        
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
        
    # If very few points, fall back to lower degree or zero
    if n < 6:
        # Fit with degree n-1 if possible, else 0
        deg = max(0, n - 1)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            # Pad to length 6
            full_coeff = np.zeros(6)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Check time
    if time.time() - t_start > deadline:
        # Fallback to linear fit if we are running out of time (unlikely for polyfit)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            full_coeff = np.zeros(6)
            full_coeff[:2] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Attempt degree 5 fit
    try:
        # polyfit returns coefficients in ascending powers
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        
        # Verify coefficients are finite
        if not np.all(np.isfinite(coeff)):
             # Fallback to linear
             coeff = np.polynomial.polynomial.polyfit(x, y, 1)
             full_coeff = np.zeros(6)
             full_coeff[:2] = coeff
             return full_coeff
             
        return coeff.astype(np.float64)
    except Exception:
        # Fallback to linear
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            full_coeff = np.zeros(6)
            full_coeff[:2] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)
