# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s wall-clock kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover unknown f(x, y, z) using a flexible 10-term basis.
    
    Basis design (10 terms):
      p0: 1 (intercept)
      p1: xc = x - 1.25
      p2: yc = y - 1.25
      p3: zc = z - 1.25
      p4: xc^2
      p5: yc^2
      p6: zc^2
      p7: xc*yc
      p8: xc*zc
      p9: yc*zc
    
    Complete quadratic in centered variables, well-conditioned for BFGS.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return zero prediction as fallback (better than crashing)
        x = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Center inputs to improve conditioning
    xc = x - 1.25
    yc = y - 1.25
    zc = z - 1.25
    
    p = params
    
    # Quadratic basis in centered variables
    pred = (p[0]
            + p[1] * xc
            + p[2] * yc
            + p[3] * zc
            + p[4] * xc * xc
            + p[5] * yc * yc
            + p[6] * zc * zc
            + p[7] * xc * yc
            + p[8] * xc * zc
            + p[9] * yc * zc)
    
    return pred
# EVOLVE-BLOCK-END
