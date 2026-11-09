import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a polynomial and tanh basis.
    
    Basis terms:
    1. Constant
    2. x
    3. v
    4. x^2
    5. v^2
    6. x*v
    7. x^3
    8. v^3
    9. tanh(x)
    10. tanh(v)
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    
    # Compute basis terms
    t0 = np.ones_like(x)
    t1 = x
    t2 = v
    t3 = x * x
    t4 = v * v
    t5 = x * v
    t6 = x * x * x
    t7 = v * v * v
    t8 = np.tanh(x)
    t9 = np.tanh(v)
    
    # Linear combination
    return (params[0] * t0 + 
            params[1] * t1 + 
            params[2] * t2 + 
            params[3] * t3 + 
            params[4] * t4 + 
            params[5] * t5 + 
            params[6] * t6 + 
            params[7] * t7 + 
            params[8] * t8 + 
            params[9] * t9)
