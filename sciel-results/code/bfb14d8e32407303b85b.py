"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled with a hybrid of
    linear base terms and adaptive RBF (Gaussian) local corrections.
    
    The linear part captures global dynamics, while RBF terms capture
    local nonlinearities. Centers are mapped through tanh to keep them
    within the data range, improving BFGS convergence.
    """
    # params layout (10 total):
    # p[0], p[1]: linear coefficients for x, v
    # p[2]: constant bias
    # p[3], p[4]: RBF center for x, v (mapped through tanh)
    # p[5]: RBF amplitude
    # p[6]: RBF width (mapped through softplus for positivity)
    # p[7], p[8]: second RBF center for x, v (mapped through tanh)
    # p[9]: second RBF amplitude
    
    # Scale inputs based on training ranges
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    x_scale = 0.55
    v_scale = 0.25
    
    X = x / x_scale
    V = v / v_scale
    
    # Linear base
    linear = params[0] * x + params[1] * v + params[2]
    
    # RBF centers mapped through tanh to stay in data range
    cx1 = np.tanh(params[3]) * x_scale
    cv1 = np.tanh(params[4]) * v_scale
    cx2 = np.tanh(params[7]) * x_scale
    cv2 = np.tanh(params[8]) * v_scale
    
    # RBF width mapped through softplus for positivity
    width = np.log1p(np.exp(params[6])) + 0.01
    
    # Compute squared distances
    dx1 = x - cx1
    dv1 = v - cv1
    dist1_sq = (dx1**2 + dv1**2) / (width**2)
    
    dx2 = x - cx2
    dv2 = v - cv2
    dist2_sq = (dx2**2 + dv2**2) / (width**2)
    
    # RBF contributions
    rbf1 = params[5] * np.exp(-dist1_sq)
    rbf2 = params[9] * np.exp(-dist2_sq)
    
    return linear + rbf1 + rbf2
# EVOLVE-BLOCK-END
