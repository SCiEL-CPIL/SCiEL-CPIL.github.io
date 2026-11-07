"""QUADRATIC POLYNOMIAL BASIS FOR E. COLI GROWTH RATE."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Quadratic Polynomial Basis with Normalized Inputs.
    
    Rationale:
    - Linear in parameters: Ensures convex optimization landscape for BFGS.
    - Normalized inputs: Improves numerical conditioning for the optimizer.
    - Term Selection:
        1. Intercept
        2-5. Linear terms (b, s, temp, pH)
        6-7. Quadratic terms for temp and pH (captures bell-shaped optima)
        8-10. Key interactions (b*temp, s*temp, b*s)
    
    Input Ranges:
    b: [1, 5]
    s: [1, 40]
    temp: [1, 12]
    pH: [1, 5]
    """
    
    # Normalize inputs to [0, 1] to ensure stable optimization
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 39.0
    t_n = (temp - 1.0) / 11.0
    p_n = (pH - 1.0) / 4.0
    
    # Clip to [0, 1] to prevent numerical issues from slight out-of-bounds
    b_n = np.clip(b_n, 0.0, 1.0)
    s_n = np.clip(s_n, 0.0, 1.0)
    t_n = np.clip(t_n, 0.0, 1.0)
    p_n = np.clip(p_n, 0.0, 1.0)
    
    # Construct Basis Terms
    # 1. Intercept
    term_0 = np.ones_like(b_n)
    
    # 2-5. Linear
    term_1 = b_n
    term_2 = s_n
    term_3 = t_n
    term_4 = p_n
    
    # 6-7. Quadratic (Curvature for Temperature and pH)
    term_5 = t_n * t_n
    term_6 = p_n * p_n
    
    # 8-10. Interactions
    term_7 = b_n * t_n   # Substrate-Temp interaction
    term_8 = s_n * t_n   # Substrate-Temp interaction
    term_9 = b_n * s_n   # Substrate-Substrate interaction
    
    # Stack into a matrix (N, 10)
    # Using np.stack is efficient for this size
    features = np.stack([term_0, term_1, term_2, term_3, term_4, 
                         term_5, term_6, term_7, term_8, term_9], axis=-1)
    
    # Linear combination: features @ params
    pred = features @ params
    
    return pred

# EVOLVE-BLOCK-END
