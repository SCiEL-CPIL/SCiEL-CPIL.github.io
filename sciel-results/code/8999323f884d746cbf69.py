"""Program for symbolic regression task using Legendre polynomials and trigonometric basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model combining Legendre polynomials (for global polynomial trends)
    and trigonometric functions (for periodic/oscillatory trends).
    
    The input x is scaled to u = x/3.0 to map [-3, 3] to [-1, 1].
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            params[0:5]  -> Coefficients for Legendre polynomials P0..P4
            params[5:10] -> Coefficients for sin(pi*u), cos(pi*u), sin(2*pi*u), cos(2*pi*u), sin(3*pi*u)
    
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for non-finite params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Scale input to [-1, 1]
    u = x / 3.0
    
    # Compute Legendre Polynomials P0 to P4
    # P0(u) = 1
    # P1(u) = u
    # P2(u) = 0.5 * (3*u^2 - 1)
    # P3(u) = 0.5 * (5*u^3 - 3*u)
    # P4(u) = 0.125 * (35*u^4 - 30*u^2 + 3)
    
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    P0 = np.ones_like(u)
    P1 = u
    P2 = 0.5 * (3.0 * u2 - 1.0)
    P3 = 0.5 * (5.0 * u3 - 3.0 * u)
    P4 = 0.125 * (35.0 * u4 - 30.0 * u2 + 3.0)
    
    # Compute Trigonometric terms
    # Frequencies scaled to match domain [-1, 1]
    # pi * u covers roughly one half-period if u is in [-1,1]? 
    # sin(pi * u) goes from -sin(pi) to sin(pi) which is 0 to 0? No.
    # u in [-1, 1]. pi*u in [-pi, pi].
    # sin(pi*u) is 0 at -1, 0, 1. Peak at -0.5, 0.5.
    # cos(pi*u) is -1 at -1, 1 at 0, -1 at 1.
    # This seems like a good basis for oscillations on the interval.
    
    T0 = np.sin(np.pi * u)
    T1 = np.cos(np.pi * u)
    T2 = np.sin(2.0 * np.pi * u)
    T3 = np.cos(2.0 * np.pi * u)
    T4 = np.sin(3.0 * np.pi * u)
    
    # Combine with parameters
    # params[0] * P0 + params[1] * P1 + ...
    
    result = (
        p[0] * P0 +
        p[1] * P1 +
        p[2] * P2 +
        p[3] * P3 +
        p[4] * P4 +
        p[5] * T0 +
        p[6] * T1 +
        p[7] * T2 +
        p[8] * T3 +
        p[9] * T4
    )
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result

# EVOLVE-BLOCK-END
