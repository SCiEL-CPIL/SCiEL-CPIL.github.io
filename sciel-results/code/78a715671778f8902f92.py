# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based model for nonlinear oscillator acceleration.
    
    Uses a Gaussian RBF kernel centered at a learned point, plus linear terms.
    params layout:
      p[0], p[1]: center coordinates (cx, cv) in scaled space
      p[2]: RBF width (sigma)
      p[3]: RBF amplitude
      p[4]: linear x coefficient
      p[5]: linear v coefficient
      p[6]: x*v interaction
      p[7]: constant offset
      p[8]: quadratic x coefficient
      p[9]: quadratic v coefficient
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Scale inputs to O(1)
    xs = x / 0.6
    vs = v / 0.3
    
    # Extract parameters
    cx = p[0]
    cv = p[1]
    sigma = np.maximum(np.abs(p[2]), 0.1)  # Ensure positive width
    amp = p[3]
    lx = p[4]
    lv = p[5]
    lxy = p[6]
    const = p[7]
    qx = p[8]
    qv = p[9]
    
    # Gaussian RBF term
    dx = xs - cx
    dv = vs - cv
    r2 = dx * dx + dv * dv
    rbf = amp * np.exp(-r2 / (2.0 * sigma * sigma))
    
    # Linear and quadratic terms
    term = (const + 
            lx * xs + 
            lv * vs + 
            lxy * xs * vs + 
            qx * xs * xs + 
            qv * vs * vs + 
            rbf)
    
    return term
# EVOLVE-BLOCK-END
