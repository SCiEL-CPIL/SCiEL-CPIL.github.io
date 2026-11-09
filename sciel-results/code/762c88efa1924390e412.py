"""Nonlinear oscillator model with local linear RBF in (x,v) and Fourier time drive."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local linear RBF model in the (x, v) phase plane with Fourier time features.
    
    The 10 parameters are allocated as:
    p0: amplitude of sin(t) drive
    p1: amplitude of cos(t) drive
    p2: RBF width (shared across 3 centers)
    p3..p9: local linear coefficients (3 centers x 3 params: slope_x, slope_v, intercept)
    
    The RBF centers are fixed at (x=0, v=0), (x=0.1, v=0.1), (x=-0.1, v=-0.1)
    to cover the observed data range. Each center provides a local linear
    approximation: w_x * (x - cx) + w_v * (v - cv) + w_0, weighted by
    exp(-||[x,v] - [cx,cv]||^2 / (2*sigma^2)).
    
    This allows the local slopes and intercepts to adapt freely to the data,
    removing structural biases of a global polynomial basis.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Fourier time features
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # RBF width: use exp(-p2) to keep it positive and bounded
    sigma2 = np.exp(-p[2]) * 0.01  # scale factor for reasonable width
    sigma2 = np.maximum(sigma2, 1e-6)  # avoid division by zero
    
    # Fixed RBF centers in (x, v) space
    # Center 1: origin
    cx1, cv1 = 0.0, 0.0
    # Center 2: positive quadrant
    cx2, cv2 = 0.1, 0.1
    # Center 3: negative quadrant
    cx3, cv3 = -0.1, -0.1
    
    # Compute RBF weights
    d1 = (x - cx1)**2 + (v - cv1)**2
    d2 = (x - cx2)**2 + (v - cv2)**2
    d3 = (x - cx3)**2 + (v - cv3)**2
    
    w1 = np.exp(-d1 / (2.0 * sigma2))
    w2 = np.exp(-d2 / (2.0 * sigma2))
    w3 = np.exp(-d3 / (2.0 * sigma2))
    
    # Local linear terms for each center
    # Center 1: p3*x + p4*v + p5
    local1 = p[3] * x + p[4] * v + p[5]
    # Center 2: p6*x + p7*v + p8
    local2 = p[6] * x + p[7] * v + p[8]
    # Center 3: p9*x + ... wait, we only have p9 left. Let me restructure.
    
    # Actually with 10 params: p0, p1 for time, p2 for width, that's 3.
    # Remaining 7 params for 3 centers. Can't do full local linear (needs 9).
    # Let's use: 3 centers, each with (slope_x, intercept) = 6 params, plus 1 shared slope_v
    # Or: use 2 centers with full local linear (6 params) + 1 bias = 7
    
    # Revised allocation:
    # p0: sin(t) amplitude
    # p1: cos(t) amplitude
    # p2: RBF width (log scale)
    # p3: center1 slope_x, p4: center1 slope_v, p5: center1 intercept
    # p6: center2 slope_x, p7: center2 slope_v, p8: center2 intercept
    # p9: global bias
    
    # Recompute with 2 centers
    # Center 1 at (0, 0)
    d1 = x**2 + v**2
    w1 = np.exp(-d1 / (2.0 * sigma2))
    
    # Center 2 at (0.1, 0.15)
    cx2, cv2 = 0.1, 0.15
    d2 = (x - cx2)**2 + (v - cv2)**2
    w2 = np.exp(-d2 / (2.0 * sigma2))
    
    # Local linear predictions
    local1 = p[3] * x + p[4] * v + p[5]
    local2 = p[6] * (x - cx2) + p[7] * (v - cv2) + p[8]
    
    # Weighted combination + global bias + time drive
    result = (w1 * local1 + w2 * local2 + p[9] +
              p[0] * sin_t + p[1] * cos_t)
    
    # Safety: replace any non-finite values with 0
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
