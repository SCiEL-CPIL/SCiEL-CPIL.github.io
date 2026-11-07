# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Basis Model: Polynomial + Learnable Gaussian Bumps + Tanh Step.
    
    Parameters (10 total):
    0: Constant bias
    1: Linear coefficient
    2: Quadratic coefficient
    3: Amplitude of Gaussian 1
    4: Center of Gaussian 1
    5: Width of Gaussian 1
    6: Amplitude of Gaussian 2
    7: Center of Gaussian 2
    8: Width of Gaussian 2
    9: Amplitude of Tanh(x)
    
    This structure allows the model to place non-linear features (bumps)
    at arbitrary locations in the domain [-2, 2] via learnable centers,
    which is a significant structural improvement over fixed Fourier/Poly bases.
    """
    p = params
    
    # Polynomial part
    t0 = np.ones_like(x)
    t1 = x
    t2 = x * x
    
    # Gaussian Bump 1
    # Using safe width to avoid division by zero
    w1 = p[5]
    w1_sq = w1 * w1
    # If w1 is extremely small, the Gaussian becomes a spike.
    # We add a small epsilon to prevent numerical issues in the exponent denominator.
    # However, BFGS might try to make w1 0. 
    # Let's just use w1_sq + 1e-6
    t3 = p[3] * np.exp(-0.5 * ((x - p[4]) ** 2) / (w1_sq + 1e-6))
    
    # Gaussian Bump 2
    w2 = p[8]
    w2_sq = w2 * w2
    t4 = p[6] * np.exp(-0.5 * ((x - p[7]) ** 2) / (w2_sq + 1e-6))
    
    # Tanh Step
    # Fixed width 1.0 for simplicity, or could be learnable if we had more params.
    # Using tanh(x) as a general non-linear saturation.
    t5 = p[9] * np.tanh(x)
    
    y = (p[0] * t0 +
         p[1] * t1 +
         p[2] * t2 +
         t3 +
         t4 +
         t5)
         
    return y
# EVOLVE-BLOCK-END
