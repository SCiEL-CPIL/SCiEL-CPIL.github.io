import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator.
    Uses a scaled polynomial basis with odd/even separation to improve conditioning.
    Inputs:
        x: array of positions
        v: array of velocities
        params: array of 10 fitted constants
    Output:
        array of accelerations
    """
    # Scale inputs to roughly unit scale to help BFGS conditioning
    # x range ~ [-0.57, 0.53] -> *2 gives ~ [-1.1, 1.0]
    # v range ~ [-0.17, 0.28] -> *5 gives ~ [-0.9, 1.4]
    xs = x * 2.0
    vs = v * 5.0
    
    # Pre-compute powers
    xs2 = xs * xs
    xs3 = xs2 * xs
    xs5 = xs3 * xs2
    
    vs2 = vs * vs
    vs3 = vs2 * vs
    vs5 = vs3 * vs2
    
    # Basis functions designed for nonlinear oscillator dynamics:
    # For a nonlinear oscillator, the restoring force is typically an odd function of x
    # and damping is typically an odd function of v (linear + nonlinear damping)
    # 
    # 0: 1 (bias/constant)
    # 1: xs (linear restoring)
    # 2: vs (linear damping)
    # 3: xs3 (cubic restoring - Duffing-like)
    # 4: vs3 (cubic damping)
    # 5: xs5 (quintic restoring)
    # 6: vs5 (quintic damping)
    # 7: xs2 * vs (nonlinear damping coupled to position)
    # 8: xs * vs2 (velocity-squared coupling to position)
    # 9: xs2 * vs2 (higher-order coupling)
    
    pred = np.zeros_like(x, dtype=float)
    
    pred += params[0] * 1.0
    pred += params[1] * xs
    pred += params[2] * vs
    pred += params[3] * xs3
    pred += params[4] * vs3
    pred += params[5] * xs5
    pred += params[6] * vs5
    pred += params[7] * (xs2 * vs)
    pred += params[8] * (xs * vs2)
    pred += params[9] * (xs2 * vs2)
    
    return pred
