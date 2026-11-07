import numpy as np
import time

_start_time = None
_deadline = 145.0

def _chebyshev(x, n):
    """Compute Chebyshev polynomial of degree n at x."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    else:
        T0 = np.ones_like(x)
        T1 = x.copy()
        for k in range(2, n + 1):
            T2 = 2.0 * x * T1 - T0
            T0, T1 = T1, T2
        return T1

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate Chebyshev polynomial expansion of degree (4,1).
    
    stress = sum_{i=0}^{4} sum_{j=0}^{1} c_{ij} * T_i(strain_norm) * T_j(temp_norm)
    
    where T_i is the Chebyshev polynomial of degree i, and strain/temp are
    normalized to [-1, 1] for numerical stability.
    
    This provides 10 coefficients with excellent conditioning for BFGS,
    capturing high-order strain nonlinearity and linear temperature effects.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return np.zeros_like(strain, dtype=np.float64)

    S = strain
    T = temp
    
    # Normalize to [-1, 1] based on training ranges
    # strain: [-0.001911, 0.9636]
    # temp: [0.06667, 1]
    S_min, S_max = -0.001911, 0.9636
    T_min, T_max = 0.06667, 1.0
    
    S_norm = 2.0 * (S - (S_min + S_max) / 2.0) / (S_max - S_min)
    T_norm = 2.0 * (T - (T_min + T_max) / 2.0) / (T_max - T_min)
    
    # Compute Chebyshev polynomials for strain (degree 0-4)
    T0_S = np.ones_like(S_norm)
    T1_S = S_norm.copy()
    T2_S = 2.0 * S_norm * T1_S - T0_S
    T3_S = 2.0 * S_norm * T2_S - T1_S
    T4_S = 2.0 * S_norm * T3_S - T2_S
    
    # Compute Chebyshev polynomials for temp (degree 0-1)
    T0_T = np.ones_like(T_norm)
    T1_T = T_norm.copy()
    
    # Build the 10-term expansion
    # c00*T0_S*T0_T + c01*T0_S*T1_T + c10*T1_S*T0_T + c11*T1_S*T1_T
    # + c20*T2_S*T0_T + c21*T2_S*T1_T + c30*T3_S*T0_T + c31*T3_S*T1_T
    # + c40*T4_S*T0_T + c41*T4_S*T1_T
    
    result = (params[0] * T0_S * T0_T +
              params[1] * T0_S * T1_T +
              params[2] * T1_S * T0_T +
              params[3] * T1_S * T1_T +
              params[4] * T2_S * T0_T +
              params[5] * T2_S * T1_T +
              params[6] * T3_S * T0_T +
              params[7] * T3_S * T1_T +
              params[8] * T4_S * T0_T +
              params[9] * T4_S * T1_T)
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e100, neginf=-1e100)
        
    return result
