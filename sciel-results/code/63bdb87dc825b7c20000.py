import numpy as np
import time

# Precompute RBF basis values for a grid of x values
# We'll use a combination of RBFs with different centers and bandwidths
# plus some global basis functions (polynomials, trig) for robustness

# Centers spread across [0.1, 6]
centers = np.linspace(0.1, 6.0, 10)
# Varying bandwidths - mix of narrow and wide
bandwidths = np.array([0.3, 0.5, 0.8, 0.3, 0.5, 0.8, 0.3, 0.5, 0.8, 1.0])

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # 10 RBF basis functions with fixed centers and bandwidths
    # Each RBF: exp(-((x - c)^2) / (2 * sigma^2))
    # We'll use all 10 parameters for the RBFs
    # But we also want some global structure. Let's use:
    # params[0:8] for 8 RBFs
    # params[8] for a linear term
    # params[9] for a constant term
    
    result = np.zeros_like(x)
    
    # 8 RBF terms
    for i in range(8):
        result += params[i] * np.exp(-((x - centers[i]) ** 2) / (2.0 * bandwidths[i] ** 2))
    
    # Linear term
    result += params[8] * x
    
    # Constant term
    result += params[9]
    
    return result
