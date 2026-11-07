import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters basis expansion for stress-strain of an aluminium rod.
    
    This model uses a set of 10 fixed basis functions of strain (epsilon) and temperature (T).
    sigma = p0*phi0 + p1*phi1 + ... + p9*phi9
    
    By making the model linear in parameters, the optimization landscape for BFGS becomes
    a convex quadratic surface, ensuring convergence to the global minimum (the least-squares
    solution) regardless of the initial parameter guess. This avoids local minima issues
    common in non-linear physics-based fits.
    
    Basis functions:
    0: 1                     (Offset)
    1: strain                (Linear elastic)
    2: strain * T            (Linear elastic, temp dependent)
    3: strain**2             (Quadratic hardening)
    4: strain**3             (Cubic hardening)
    5: strain**5             (High order hardening)
    6: strain**2 * T         (Quadratic hardening, temp dependent)
    7: strain**3 * T         (Cubic hardening, temp dependent)
    8: T                     (Temp offset)
    9: T**2                  (Quadratic temp offset)
    """
    p = params
    
    # Ensure inputs are float arrays
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    # If parameters are not finite, return zeros to avoid propagation of NaNs/Infs
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
        
    # Pre-compute powers of strain and temp
    # Using np.power or ** is fine. 
    # We include a small epsilon in the power calculation? No, integer powers are safe.
    # If strain can be negative, odd powers are negative, even are positive.
    # This allows the model to learn asymmetric behavior if the data is asymmetric,
    # although stress-strain is usually symmetric. The linear combination allows 
    # cancellation if needed.
    
    s = strain
    t = temp
    
    s2 = s * s
    s3 = s2 * s
    s5 = s3 * s2
    
    t2 = t * t
    
    # Construct basis matrix features
    # phi0: 1
    phi0 = 1.0
    
    # phi1: s
    phi1 = s
    
    # phi2: s * t
    phi2 = s * t
    
    # phi3: s^2
    phi3 = s2
    
    # phi4: s^3
    phi4 = s3
    
    # phi5: s^5
    phi5 = s5
    
    # phi6: s^2 * t
    phi6 = s2 * t
    
    # phi7: s^3 * t
    phi7 = s3 * t
    
    # phi8: t
    phi8 = t
    
    # phi9: t^2
    phi9 = t2
    
    # Compute linear combination
    sigma = (p[0] * phi0 +
             p[1] * phi1 +
             p[2] * phi2 +
             p[3] * phi3 +
             p[4] * phi4 +
             p[5] * phi5 +
             p[6] * phi6 +
             p[7] * phi7 +
             p[8] * phi8 +
             p[9] * phi9)
             
    return sigma
