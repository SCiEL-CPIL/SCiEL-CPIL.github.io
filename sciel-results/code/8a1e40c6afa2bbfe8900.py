import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses scaled inputs to improve numerical conditioning for BFGS.
    Basis includes linear, quadratic, and cubic terms, plus a quartic term for x.
    """
    p = params
    
    # Scale inputs to O(1) range to help optimizer convergence
    # x range approx [-0.57, 0.53], v range approx [-0.18, 0.28]
    xs = x / 0.6
    vs = v / 0.3
    
    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2
    vs2 = vs * vs
    
    # Construct basis functions
    # 1. Constant
    # 2. Linear x
    # 3. Linear v
    # 4. Quadratic x
    # 5. Cubic x (Duffing nonlinearity)
    # 6. Quadratic v
    # 7. Interaction x*v
    # 8. Interaction x^2*v
    # 9. Interaction x*v^2
    # 10. Quartic x
    
    return (
        p[0] + 
        p[1] * xs + 
        p[2] * vs + 
        p[3] * xs2 + 
        p[4] * xs3 + 
        p[5] * vs2 + 
        p[6] * xs * vs + 
        p[7] * xs2 * vs + 
        p[8] * xs * vs2 + 
        p[9] * xs4
    )
