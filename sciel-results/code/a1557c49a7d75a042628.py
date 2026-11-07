# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a Gaussian RBF network with five fixed centers.
    
    Structure:
    f(x, y, z) = p0 + sum_{i=1}^{5} p_i * exp(-w^2 * ||(x,y,z) - c_i||^2)
    
    where c_i are five fixed centers derived from the input domain [0.5, 2]^3:
      c_1 = (0.5, 0.5, 0.5)
      c_2 = (2.0, 2.0, 2.0)
      c_3 = (1.25, 1.25, 1.25)
      c_4 = (0.5, 2.0, 1.25)
      c_5 = (2.0, 0.5, 1.25)
    
    Parameters:
      p0: global bias
      p1..p5: amplitudes for each RBF
      w: shared width scale (param index 6)
      p7..p9: linear correction terms for x, y, z to capture global trends
    
    This local approximation strategy can fit complex, non-separable surfaces
    that are orthogonal to global rational or multiplicative forms.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    if params.size < 10:
        params = np.concatenate([params, np.zeros(10 - params.size)])
    
    # Extract parameters
    p0 = params[0]  # bias
    p1, p2, p3, p4, p5 = params[1], params[2], params[3], params[4], params[5]  # RBF amplitudes
    w = params[6]   # width scale
    p7, p8, p9 = params[7], params[8], params[9]  # linear corrections
    
    # Fixed centers
    c1 = np.array([0.5, 0.5, 0.5])
    c2 = np.array([2.0, 2.0, 2.0])
    c3 = np.array([1.25, 1.25, 1.25])
    c4 = np.array([0.5, 2.0, 1.25])
    c5 = np.array([2.0, 0.5, 1.25])
    
    # Compute squared distances to each center
    # d1^2 = (x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2
    d1_sq = (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2
    d2_sq = (x - 2.0)**2 + (y - 2.0)**2 + (z - 2.0)**2
    d3_sq = (x - 1.25)**2 + (y - 1.25)**2 + (z - 1.25)**2
    d4_sq = (x - 0.5)**2 + (y - 2.0)**2 + (z - 1.25)**2
    d5_sq = (x - 2.0)**2 + (y - 0.5)**2 + (z - 1.25)**2
    
    # Width scale: use abs to ensure positive, with a minimum to prevent overflow
    w_safe = np.maximum(np.abs(w), 1e-6)
    w_sq = w_safe ** 2
    
    # Compute RBF values
    # Clip the exponent argument to prevent overflow/underflow issues
    # For large negative exponents, exp() -> 0, which is fine
    # For large positive exponents (shouldn't happen with w_sq > 0 and d_sq >= 0), clip
    arg1 = -w_sq * d1_sq
    arg2 = -w_sq * d2_sq
    arg3 = -w_sq * d3_sq
    arg4 = -w_sq * d4_sq
    arg5 = -w_sq * d5_sq
    
    # Clip to avoid numerical issues (exp of very large positive -> inf)
    # Since w_sq > 0 and d_sq >= 0, args are <= 0, so exp is in (0, 1]
    # But just in case of any numerical edge cases:
    arg1 = np.minimum(arg1, 0.0)
    arg2 = np.minimum(arg2, 0.0)
    arg3 = np.minimum(arg3, 0.0)
    arg4 = np.minimum(arg4, 0.0)
    arg5 = np.minimum(arg5, 0.0)
    
    rbf1 = np.exp(arg1)
    rbf2 = np.exp(arg2)
    rbf3 = np.exp(arg3)
    rbf4 = np.exp(arg4)
    rbf5 = np.exp(arg5)
    
    # Combine
    pred = (p0 +
            p1 * rbf1 +
            p2 * rbf2 +
            p3 * rbf3 +
            p4 * rbf4 +
            p5 * rbf5 +
            p7 * x +
            p8 * y +
            p9 * z)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
