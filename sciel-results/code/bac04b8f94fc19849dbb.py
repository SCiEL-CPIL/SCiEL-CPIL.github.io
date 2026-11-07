# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fixed 2-layer tanh network.
    
    Architecture:
    - Input: (x, y)
    - Hidden Layer 1: 12 tanh units
    - Hidden Layer 2: 9 tanh units + 1 constant unit (for bias)
    - Output: Linear combination of Hidden Layer 2 units using 10 parameters.
    
    The internal weights are fixed using a deterministic pseudo-random seed.
    The 10 parameters are the weights of the final linear layer.
    This structure is linear in parameters, making it ideal for BFGS optimization.
    """
    t0 = time.time()
    
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    
    # Parameters
    p = np.asarray(params, dtype=float).ravel()
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size))
    else:
        p = p[:10]

    # Fixed internal weights
    # Using a fixed seed to ensure the basis is consistent across evaluations
    # but "random" enough to span the function space.
    rng = np.random.RandomState(42)
    
    # Hidden Layer 1
    # Inputs: x, y, 1 (bias)
    # Weights: 12 units x 3 inputs
    # Scale: Inputs are in [-1, 1]. We want arguments to tanh to be in a range
    # where the function is sensitive. [-3, 3] is good.
    # Variance of x+y+1 is approx 1. So weights should be O(1).
    W1 = rng.uniform(-2.0, 2.0, size=(12, 3))
    b1 = rng.uniform(-1.0, 1.0, size=(12, 1))
    
    # Hidden Layer 2
    # Inputs: 12 units from H1
    # Weights: 9 units x 12 inputs (we only need 9 non-constant units)
    W2 = rng.uniform(-2.0, 2.0, size=(9, 12))
    b2 = rng.uniform(-1.0, 1.0, size=(9, 1))
    
    # Compute Hidden Layer 1
    # Inputs matrix: (n, 3)
    inputs1 = np.column_stack((x, y, np.ones(n)))
    z1 = inputs1 @ W1.T + b1.T  # (n, 12)
    h1 = np.tanh(z1)
    
    # Compute Hidden Layer 2 (non-constant part)
    z2 = h1 @ W2.T + b2.T  # (n, 9)
    h2_tanh = np.tanh(z2)
    
    # Construct final basis matrix
    # Basis has 10 columns: 9 from h2_tanh, 1 constant
    B = np.empty((n, 10), dtype=float)
    B[:, 0:9] = h2_tanh
    B[:, 9] = 1.0
    
    # Final prediction
    out = B @ p
    
    # Safety checks
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Deadline guard
    if time.time() - t0 > 140.0:
        return out
        
    return out
# EVOLVE-BLOCK-END
