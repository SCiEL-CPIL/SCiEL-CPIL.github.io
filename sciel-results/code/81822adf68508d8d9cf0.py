"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for safe operations, though it usually is
    # x shape: (N,)
    
    # Compute basis functions
    # We use 10 basis functions to match the 10 parameters
    
    # Scale x to [-1, 1] for polynomial stability
    t = x / 3.0
    
    # 1. Constant
    f0 = np.ones_like(x)
    # 2. Linear (Scaled)
    f1 = t
    # 3. Quadratic (Chebyshev T2)
    f2 = 2 * t**2 - 1
    # 4. Cubic (Chebyshev T3)
    f3 = 4 * t**3 - 3 * t
    # 5. Quartic (Chebyshev T4)
    f4 = 8 * t**4 - 8 * t**2 + 1
    # 6. Sine
    f5 = np.sin(x)
    # 7. Cosine
    f6 = np.cos(x)
    # 7. Sine 2x
    f7 = np.sin(2 * x)
    # 8. Cosine 2x
    f8 = np.cos(2 * x)
    # 9. Tanh
    f9 = np.tanh(x)
    
    # Stack features into a matrix of shape (N, 10)
    # Columns correspond to params[0] ... params[9]
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    # Linear combination
    # params shape: (10,)
    # features shape: (N, 10)
    # Result shape: (N,)
    return features.dot(params)

# EVOLVE-BLOCK-END
