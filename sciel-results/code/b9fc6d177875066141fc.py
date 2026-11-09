import numpy as np

# Domain: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with Gaussian kernels.
    
    Uses 10 parameters to define 3 RBF basis functions plus a bias:
    - 3 basis functions, each with: center_x, center_v, width, weight
    - 1 bias term (constant offset)
    
    Each RBF: weight_i * exp(-((x - cx_i)^2 + (v - cv_i)^2) / (2 * w_i^2))
    
    Parameters are mapped from [0, 1] (BFGS starts at 1) to meaningful ranges:
    - center_x: mapped to [x_min, x_max]
    - center_v: mapped to [v_min, v_max]
    - width: mapped to [0.01, 1.0] (in normalized units)
    - weight: mapped to [-10, 10]
    """
    # Normalize inputs to [-1, 1] range for better conditioning
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale

    # Extract parameters for 3 RBFs + bias
    # params[0] = bias
    # params[1:5] = RBF 1 (cx, cv, w, weight)
    # params[5:9] = RBF 2 (cx, cv, w, weight)
    # params[9]   = RBF 3 weight (with cx, cv, w fixed at 0 for stability)

    bias = params[0]

    # For RBF 1 and 2, use all 4 params
    # Map center params: assume params are in [-1, 1] range after optimization
    # Use tanh to keep centers bounded, then scale to input range
    
    # RBF 1
    cx1 = np.tanh(params[1]) * 0.9  # keep within [-0.9, 0.9]
    cv1 = np.tanh(params[2]) * 0.9
    w1 = np.exp(params[3]) * 0.5 + 0.05  # width in [0.05, ~0.5]
    wt1 = np.tanh(params[4]) * 5.0  # weight in [-5, 5]

    # RBF 2
    cx2 = np.tanh(params[5]) * 0.9
    cv2 = np.tanh(params[6]) * 0.9
    w2 = np.exp(params[7]) * 0.5 + 0.05
    wt2 = np.tanh(params[8]) * 5.0

    # RBF 3: fixed center at origin, use last param as weight
    cx3 = 0.0
    cv3 = 0.0
    w3 = 0.3  # fixed width
    wt3 = np.tanh(params[9]) * 5.0

    # Compute Gaussian RBFs
    # Use squared distance for stability
    dx1 = xn - cx1
    dy1 = vn - cv1
    r1 = dx1 * dx1 + dy1 * dy1
    rbf1 = np.exp(-r1 / (2.0 * w1 * w1))

    dx2 = xn - cx2
    dy2 = vn - cv2
    r2 = dx2 * dx2 + dy2 * dy2
    rbf2 = np.exp(-r2 / (2.0 * w2 * w2))

    dx3 = xn - cx3
    dy3 = vn - cv3
    r3 = dx3 * dx3 + dy3 * dy3
    rbf3 = np.exp(-r3 / (2.0 * w3 * w3))

    # Linear combination
    result = bias + wt1 * rbf1 + wt2 * rbf2 + wt3 * rbf3

    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)

    return result
