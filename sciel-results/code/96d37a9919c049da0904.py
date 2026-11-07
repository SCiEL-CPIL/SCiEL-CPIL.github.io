# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Pure bivariate polynomial chaos expansion in strain and temperature.
    
    This model uses a linear combination of 10 monomial basis functions:
    1, s, t, s^2, st, t^2, s^3, s^2t, st^2, s^3t
    
    where s = strain (centered/scaled for numerical conditioning) and t = temp (centered/scaled).
    
    Being linear in the parameters, the optimization landscape for BFGS is convex,
    avoiding local minima issues associated with non-linear constitutive models.
    The bivariate interaction terms allow capturing coupled strain-temperature effects
    without the structural bias of physical laws.
    
    params: [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return zeros if we are getting close to the 150s limit
    # The evaluator kills at 150s. We want to be safe.
    # Since this function is called many times during fitting, we check time.
    # However, checking time every call is overhead. 
    # The instruction says: "Keep an EXPLICIT deadline guard".
    # If elapsed > 140s, return zeros to stop computation.
    elapsed = time.time() - _start_time
    if elapsed > 140.0:
        return np.zeros_like(strain, dtype=np.float64)
    
    s = np.asarray(strain, dtype=np.float64)
    t = np.asarray(temp, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # If any param is non-finite, return zeros
    if not np.all(np.isfinite(p)):
        return np.zeros_like(s)
        
    # Conditioning:
    # Strain range: [-0.001911, 0.9636]. Center ~ 0.48, Range ~ 0.965.
    # Temp range: [0.06667, 1]. Center ~ 0.53, Range ~ 0.93.
    # We scale to roughly [-1, 1] to improve numerical conditioning of high-order terms.
    # s_norm = (s - 0.48) / 0.5
    # t_norm = (t - 0.53) / 0.5
    # Using fixed normalization constants based on training ranges.
    
    s_mean = 0.48
    s_scale = 0.5
    t_mean = 0.53
    t_scale = 0.5
    
    sn = (s - s_mean) / s_scale
    tn = (t - t_mean) / t_scale
    
    # Basis functions
    # Order:
    # 0: 1
    # 1: sn
    # 2: tn
    # 3: sn^2
    # 4: sn*tn
    # 5: tn^2
    # 6: sn^3
    # 7: sn^2*tn
    # 8: sn*tn^2
    # 9: sn^3*tn
    
    b0 = np.ones_like(sn)
    b1 = sn
    b2 = tn
    b3 = sn * sn
    b4 = sn * tn
    b5 = tn * tn
    b6 = b3 * sn
    b7 = b3 * tn
    b8 = b1 * b5
    b9 = b6 * tn
    
    # Linear combination
    result = (p[0] * b0 + 
              p[1] * b1 + 
              p[2] * b2 + 
              p[3] * b3 + 
              p[4] * b4 + 
              p[5] * b5 + 
              p[6] * b6 + 
              p[7] * b7 + 
              p[8] * b8 + 
              p[9] * b9)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
