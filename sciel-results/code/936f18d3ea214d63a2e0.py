import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid symbolic regression model combining:
    1. Chebyshev polynomial basis (T1-T4) for smooth approximation
    2. Rational function component (cubic/cubic) for flexible ratio forms
    3. Exponential component with learned rate
    4. Logarithmic component
    
    Structure:
    y = c0 + c1*T1 + c2*T2 + c3*T3 + c4*T4
        + (p0 + p1*u + p2*u^2 + p3*u^3) / (1 + p4*u + p5*u^2 + p6*u^3)
        + p7 * exp(p8 * u)
        + p9 * log(x + 1)
    
    Where u = (x - 3.05) / 2.95 (standardized to approx [-1, 1])
    
    Parameters (10 total):
    c0-c4: Chebyshev coefficients (indices 0-4 in params)
    p0-p6: Rational numerator/denominator coefficients (indices 5-11 mapped from params)
    p7: Exponential amplitude (index 8)
    p8: Exponential rate (index 9)
    p9: Logarithmic coefficient (index 9 shared)
    
    Wait - we only have 10 params. Let me redesign:
    
    New allocation:
    params[0]: Chebyshev bias (T0 coefficient)
    params[1]: Chebyshev T1 coefficient
    params[2]: Chebyshev T2 coefficient
    params[3]: Rational numerator p0
    params[4]: Rational numerator p1
    params[5]: Rational denominator p4 (for u term)
    params[6]: Rational denominator p5 (for u^2 term)
    params[7]: Exponential amplitude
    params[8]: Exponential rate
    params[9]: Logarithmic coefficient
    
    This gives us a good mix of linear-in-params smooth terms plus nonlinear rational/exp/log.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Ensure p has length 10
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=1.0)
    else:
        p = p[:10]
    
    # Standardize x: map [0.1, 6] to approx [-1, 1]
    x_center = 3.05
    x_scale = 2.95
    u = (x - x_center) / x_scale
    
    # Clip u to prevent extreme values
    u = np.clip(u, -2.0, 2.0)
    
    # Compute Chebyshev polynomials on u (already in [-1,1] range mostly)
    T0 = np.ones_like(x)
    T1 = u
    T2 = 2.0 * u**2 - 1.0
    T3 = 4.0 * u**3 - 3.0 * u
    
    # Chebyshev part: p[0]*T0 + p[1]*T1 + p[2]*T2 + p[3]*T3
    # But wait, I allocated p[3] to rational. Let me restructure.
    
    # Revised allocation for 10 params:
    # p[0]: Bias / Chebyshev T0 coeff
    # p[1]: Chebyshev T1 coeff
    # p[2]: Chebyshev T2 coeff
    # p[3]: Chebyshev T3 coeff
    # p[4]: Rational numerator linear coeff (u term)
    # p[5]: Rational numerator quadratic coeff (u^2 term)
    # p[6]: Rational denominator linear coeff
    # p[7]: Exponential amplitude
    # p[8]: Exponential rate
    # p[9]: Logarithmic coefficient
    
    # Chebyshev part
    cheb_part = p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3
    
    # Rational part: (a0 + a1*u + a2*u^2) / (1 + b1*u + b2*u^2)
    # Use p[4], p[5] for numerator (u and u^2 terms), p[6] for denominator (u term)
    # Keep it simple: numerator = 1 + p[4]*u + p[5]*u^2, denominator = 1 + p[6]*u
    u2 = u * u
    num = 1.0 + p[4] * u + p[5] * u2
    den = 1.0 + p[6] * u
    
    # Safety: Ensure denominator is not close to zero
    den_safe = np.where(np.abs(den) < 1e-6, 1e-6, den)
    
    rational_part = num / den_safe
    
    # Exponential part: p[7] * exp(p[8] * u)
    exp_arg = p[8] * u
    exp_arg = np.clip(exp_arg, -50.0, 50.0)
    exp_part = p[7] * np.exp(exp_arg)
    
    # Logarithmic part: p[9] * log(x + 1)
    x_safe = np.clip(x, 1e-10, None)
    log_part = p[9] * np.log(x_safe + 1.0)
    
    # Combine all parts
    result = cheb_part + rational_part + exp_part + log_part
    
    # Final safety check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
