import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models the acceleration of a nonlinear oscillator using a complete 3rd-order
    polynomial in position (x) and velocity (v). This structure avoids explicit
    time-dependence which can lead to overfitting, focusing instead on the
    state-space dynamics.
    """
    # Ensure inputs are arrays
    x = np.asarray(x)
    v = np.asarray(v)
    
    # Parameters:
    # p[0]: constant
    # p[1]: linear x
    # p[2]: linear v
    # p[3]: quadratic x^2
    # p[4]: cross term x*v
    # p[5]: quadratic v^2
    # p[6]: cubic x^3
    # p[7]: cubic x^2*v
    # p[8]: cubic x*v^2
    # p[9]: cubic v^3
    
    p = params
    
    # Precompute powers to avoid redundant calculations
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    
    # Linear combination
    # Note: We ignore t based on the finding that removing time-dependent terms improves performance.
    pred = (p[0] + 
            p[1] * x + 
            p[2] * v + 
            p[3] * x2 + 
            p[4] * (x * v) + 
            p[5] * v2 + 
            p[6] * x3 + 
            p[7] * (x2 * v) + 
            p[8] * (x * v2) + 
            p[9] * v3)
            
    return pred
