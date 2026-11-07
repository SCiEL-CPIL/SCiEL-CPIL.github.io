import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a sum of Gaussian Radial Basis Functions (RBFs).
    This approach is robust for BFGS because:
    1. RBFs are smooth and localized.
    2. We use parameter offsets to ensure the initial centers (when params are all 1)
       are spread across the domain [-2, 2], avoiding the "all centers at same spot"
       local minimum trap.
    3. Fixed width (sigma=0.5) is appropriate for the domain size.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    original_shape = x.shape
    x_flat = x.flatten()
    N = len(x_flat)
    
    if N == 0:
        return np.empty(original_shape)
        
    p = params.flatten()
    
    # 5 Amplitudes, 5 Centers
    a = p[0:5]
    c_raw = p[5:10]
    
    # Offsets to spread initial centers when p=1
    # Initial centers: -1, 0, 1, 2, 3
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = c_raw + offsets
    
    # Fixed width sigma = 0.5
    # Exponent: -((x-c)^2) / (2 * sigma^2) = -((x-c)^2) / 0.5 = -2 * (x-c)^2
    y = np.zeros(N, dtype=float)
    
    for i in range(5):
        diff = x_flat - c[i]
        # Check for deadline inside loop? No, N is small (200).
        # Just compute.
        y += a[i] * np.exp(-2.0 * diff * diff)
        
    return y.reshape(original_shape)
