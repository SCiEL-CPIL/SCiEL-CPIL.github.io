import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 120.0 # seconds
    
    # Convert to numpy arrays
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        # Fallback to linear if data is too sparse
        coeff = np.zeros(6)
        if len(x) >= 2:
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                pass
        return coeff

    # Check time
    if time.time() > deadline - 5:
        # Fallback to linear if running out of time
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
        return coeff

    # Scale x to [-1, 1] for Legendre fit stability
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Constant x, return constant fit
        coeff = np.zeros(6)
        coeff[0] = np.mean(y)
        return coeff

    # s = (2 * x - (x_max + x_min)) / x_range
    # This maps [x_min, x_max] to [-1, 1]
    s = (2.0 * x - (x_max + x_min)) / x_range
    
    # Fit Legendre polynomial of degree 5
    # legfit returns coefficients in ascending powers of s
    try:
        leg_coeffs = np.polynomial.legende.legfit(s, y, 5)
    except Exception:
        # Fallback to linear if fit fails
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
        return coeff

    # Check time
    if time.time() > deadline - 2:
        # Fallback: use the linear fit in the scaled domain? No, we need monomial in x.
        # Just return a simple linear fit in x to be safe.
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
        return coeff

    # Convert Legendre coefficients to a Legendre polynomial object
    # The Legendre class expects coefficients in ascending order of degree
    leg_poly = np.polynomial.legende.Legendre(leg_coeffs)
    
    # Convert to monomial polynomial in s
    # P_s(s) = sum c_i s^i
    mono_s_poly = leg_poly.convert(kind=np.polynomial.polynomial.Polynomial)
    
    # Now we need to express this in terms of x.
    # s = a * x + b
    a = 2.0 / x_range
    b = -(x_max + x_min) / x_range
    
    # Create the linear polynomial for the transformation s(x)
    # s = a*x + b
    transform_poly = np.polynomial.polynomial.Polynomial([b, a])
    
    # Compose P_s(s(x))
    # mono_s_poly is P(s). We want P(a*x + b).
    # Polynomial composition: P(transform_poly)
    final_poly = mono_s_poly.compose(transform_poly)
    
    # Extract coefficients
    # final_poly.coef contains coefficients in ascending powers of x
    coeff = np.zeros(6)
    # Ensure we only take up to degree 5
    num_coeffs = min(6, len(final_poly.coef))
    coeff[:num_coeffs] = final_poly.coef[:num_coeffs]
    
    # Check for any non-finite results
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
    
    return coeff
