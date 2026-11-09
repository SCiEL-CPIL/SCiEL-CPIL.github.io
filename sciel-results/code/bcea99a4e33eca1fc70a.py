"""Rewritten program for the suite task; improved polynomial fitting."""
# EVOLVE-BLOCK-START
import numpy as np

def construct(training_x, training_subgrid_tendency):
    """
    Fits a degree-five polynomial to the subgrid tendency as a function of the slow state x.
    
    Args:
        training_x: Array-like of slow state values.
        training_subgrid_tendency: Array-like of corresponding subgrid tendencies.
        
    Returns:
        np.ndarray: Coefficients [c0, c1, c2, c3, c4, c5] for the polynomial 
                    sum(c_i * x**i).
    """
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    if len(x) == 0:
        return np.zeros(6)
        
    # Ensure we have enough points for a degree 5 fit (need at least 6, but more is better)
    # If we have very few points, we might need to reduce degree, but the task 
    # specifies a degree-five polynomial interface. We will assume sufficient data.
    # For robustness, if len(x) < 6, we fall back to lower degree or mean.
    
    n = len(x)
    degree = 5
    
    if n < degree + 1:
        # Fallback: if not enough data for degree 5, use as high a degree as possible
        # but the interface expects 6 coefficients. We will pad with zeros.
        effective_degree = min(degree, n - 1)
        if effective_degree < 0:
            return np.zeros(6)
    else:
        effective_degree = degree

    # Normalize x to improve conditioning of the polynomial fit
    x_mean = np.mean(x)
    x_std = np.std(x)
    
    if x_std < 1e-12:
        # If x is constant, the polynomial is just a constant (or undefined slope)
        # Return mean of y as c0
        coeff_x = np.zeros(6)
        coeff_x[0] = np.mean(y)
        return coeff_x

    z = (x - x_mean) / x_std
    
    # Fit polynomial in z: y = sum(a_i * z**i)
    # Using polyfit which is stable for small degrees
    try:
        # full=False returns only the coefficients
        a = np.polynomial.polynomial.polyfit(z, y, effective_degree)
    except Exception:
        # Fallback to linear if polyfit fails for some reason
        try:
            a = np.polynomial.polynomial.polyfit(z, y, 1)
            # Pad with zeros to length 6
            a_padded = np.zeros(6)
            a_padded[:len(a)] = a
            a = a_padded
            effective_degree = 1
        except Exception:
            # Last resort
            return np.zeros(6)

    # Ensure a has length 6
    if len(a) < 6:
        a_padded = np.zeros(6)
        a_padded[:len(a)] = a
        a = a_padded

    # Transform coefficients from z-basis to x-basis
    # z = (x - x_mean) / x_std = (1/x_std) * x - (x_mean/x_std)
    # Let alpha = 1/x_std, beta = -x_mean/x_std
    # z = alpha * x + beta
    # z^k = (alpha * x + beta)^k = sum_{j=0}^k binom(k, j) (alpha * x)^j beta^(k-j)
    #     = sum_{j=0}^k binom(k, j) alpha^j beta^(k-j) x^j
    
    alpha = 1.0 / x_std
    beta = -x_mean / x_std
    
    coeff_x = np.zeros(6)
    
    for k in range(effective_degree + 1):
        a_k = a[k]
        if a_k == 0:
            continue
        # Expand (alpha * x + beta)^k
        for j in range(k + 1):
            # Term for x^j
            # Coefficient is a_k * binom(k, j) * alpha^j * beta^(k-j)
            binom_k_j = 1.0
            for m in range(j):
                binom_k_j *= (k - m) / (m + 1)
            
            term = a_k * binom_k_j * (alpha ** j) * (beta ** (k - j))
            coeff_x[j] += term

    return coeff_x

# EVOLVE-BLOCK-END
