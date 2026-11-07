"""Program using a hybrid RBF-MLP basis with 5 fixed centers and learned widths."""
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    
    Uses a hybrid basis combining:
    - 2 global polynomial terms (constant, linear)
    - 5 localized Gaussian RBF kernels with fixed centers and learned widths
    
    The 10 params are:
    0-1: weights for constant and linear terms
    2-6: amplitudes for the 5 RBF kernels
    7-9: width parameters (shared or per-group) for the RBF kernels
    
    Actually, let's use:
    0-1: global weights (const, linear)
    2-6: RBF amplitudes (5 kernels)
    7-9: width parameters - we'll use 3 widths, applying them cyclically to the 5 kernels
    Or better: use a single shared width parameter to keep it simple?
    
    Let's reconsider: 10 params total.
    Option A: 2 global + 5 amplitudes + 3 widths (cyclic)
    Option B: 2 global + 4 amplitudes + 4 widths (for 4 kernels, drop one)
    Option C: 2 global + 5 amplitudes + 3 widths where widths are for groups
    
    Let's go with:
    0-1: global (const, linear)
    2-6: 5 RBF amplitudes
    7-9: 3 width parameters, applied to kernels as:
         kernel 0,1 use width 7
         kernel 2,3 use width 8
         kernel 4 uses width 9
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(x)
    
    u = x / 3.0  # Scale to [-1, 1]
    
    # Global terms
    b0 = np.ones_like(u)
    b1 = u
    
    # 5 fixed evenly spaced centers in [-1, 1]
    centers = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    
    # Width parameters from params[7:10]
    # Apply cyclically: kernels 0,1 -> w0; kernels 2,3 -> w1; kernel 4 -> w2
    w0 = np.clip(params[7], 0.05, 1.0)  # Clip to avoid degenerate widths
    w1 = np.clip(params[8], 0.05, 1.0)
    w2 = np.clip(params[9], 0.05, 1.0)
    
    # Compute RBF kernels
    # Kernel 0: center=-0.8, width=w0
    rbf0 = np.exp(-0.5 * ((u - centers[0]) ** 2) / (w0 * w0))
    # Kernel 1: center=-0.4, width=w0
    rbf1 = np.exp(-0.5 * ((u - centers[1]) ** 2) / (w0 * w0))
    # Kernel 2: center=0.0, width=w1
    rbf2 = np.exp(-0.5 * ((u - centers[2]) ** 2) / (w1 * w1))
    # Kernel 3: center=0.4, width=w1
    rbf3 = np.exp(-0.5 * ((u - centers[3]) ** 2) / (w1 * w1))
    # Kernel 4: center=0.8, width=w2
    rbf4 = np.exp(-0.5 * ((u - centers[4]) ** 2) / (w2 * w2))
    
    # Amplitudes from params[2:7]
    a0 = params[2]
    a1 = params[3]
    a2 = params[4]
    a3 = params[5]
    a4 = params[6]
    
    # Global weights
    g0 = params[0]
    g1 = params[1]
    
    # Linear combination
    y = g0 * b0 + g1 * b1 + a0 * rbf0 + a1 * rbf1 + a2 * rbf2 + a3 * rbf3 + a4 * rbf4
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return y
