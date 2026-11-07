import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a Rational Function basis.
    
    y(x) = (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) / (1 + b1*x + b2*x^2 + b3*x^3 + b4*x^4 + b5*x^5)
    
    This structure allows for modeling poles and asymptotes, which 
    polynomial and RBF bases may struggle with. The non-linear 
    dependence on parameters provides greater expressive power 
    for a wider variety of hidden functions.
    """
    x = np.asarray(x, dtype=float)
    
    # Unpack parameters
    # a0..a4 (5 params)
    # b1..b5 (5 params)
    a = params[:5]
    b = params[5:]
    
    # Compute powers of x for efficiency
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    
    # Numerator (Degree 4)
    num = a[0] + a[1]*x + a[2]*x2 + a[3]*x3 + a[4]*x4
    
    # Denominator (Degree 5, constant term fixed at 1)
    den = 1.0 + b[0]*x + b[1]*x2 + b[2]*x3 + b[3]*x4 + b[4]*x5
    
    # Safety check for division by zero
    # If den is close to zero, replace with small value to avoid inf/nan
    # This prevents crashes but may slightly bias the fit near singularities
    min_den = 1e-8
    den_safe = np.where(np.abs(den) < min_den, min_den * np.sign(den + 1e-12), den)
    
    y = num / den_safe
    
    return y
