import numpy as np

def _halton_sequence(index, base):
    """Generate the Halton sequence value for a given index and base."""
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r

def _get_centers():
    """Generate 9 fixed centers in 4D space using Halton sequence."""
    centers = []
    for i in range(1, 10):
        c = [
            _halton_sequence(i, 2),
            _halton_sequence(i, 3),
            _halton_sequence(i, 5),
            _halton_sequence(i, 7)
        ]
        centers.append(c)
    return np.array(centers)

# Pre-compute centers
_CENTERS = _get_centers()
_GAMMA = 10.0 # Inverse squared width. 10 means sigma^2 = 1/10, sigma ~ 0.31

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model for E. coli growth rate.
    Inputs are normalized to [0, 1].
    9 RBF terms + 1 bias.
    """
    # Normalize inputs to [0, 1]
    # Ranges from task description:
    # b: [1, 4.998] -> approx [1, 5]
    # s: [1, 39.98] -> approx [1, 40]
    # temp: [1, 12]
    # pH: [1, 4.999] -> approx [1, 5]
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Clip to [0, 1] to handle slight out-of-range inputs robustly (though task says in-range)
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)
    
    # Stack inputs: (N, 4)
    X = np.column_stack((b_norm, s_norm, temp_norm, ph_norm))
    
    # Weights: params[0:9] are RBF weights, params[9] is bias
    weights = params[:9]
    bias = params[9]
    
    # Compute squared distances to centers: (N, 9)
    # Centers shape: (9, 4)
    # X shape: (N, 4)
    # dist_sq = sum((X - C)^2, axis=1)
    
    # Efficient calculation:
    # ||X - C||^2 = ||X||^2 + ||C||^2 - 2 X C^T
    X_norm_sq = np.sum(X * X, axis=1, keepdims=True) # (N, 1)
    C_norm_sq = np.sum(_CENTERS * _CENTERS, axis=1, keepdims=True).T # (1, 9)
    XC = X @ _CENTERS.T # (N, 9)
    
    dist_sq = X_norm_sq + C_norm_sq - 2 * XC
    # Ensure non-negative due to floating point errors
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # Compute RBF values: exp(-gamma * dist_sq)
    rbf_vals = np.exp(-_GAMMA * dist_sq) # (N, 9)
    
    # Linear combination
    y = rbf_vals @ weights + bias
    
    return y
