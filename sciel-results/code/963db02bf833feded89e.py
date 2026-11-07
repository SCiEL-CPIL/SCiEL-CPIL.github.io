import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters Radial Basis Function model with fixed centers.
    
    The model uses 10 Gaussian RBFs with centers evenly spaced across the
    input domain [0.1, 6]. The parameters are the weights for each RBF.
    This is linear in parameters, ensuring BFGS converges reliably to the
    optimal least-squares solution.
    
    Parameters (10 total):
    p[0]: weight for RBF centered at ~0.76
    p[1]: weight for RBF centered at ~1.41
    ...
    p[9]: weight for RBF centered at ~5.24
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.ones_like(x) * (p[0] if p.size > 0 else 0.0)
    
    # Define fixed centers and width
    # Centers evenly spaced in [0.1, 6.0]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Width (sigma) controls the spread. 
    # A sigma of ~0.6-0.8 allows local fitting without too much smoothing.
    sigma = 0.7
    
    # Compute RBF values: (N, 10)
    # diff is (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: (N, 10) @ (10,) -> (N,)
    result = rbf_values @ p
    
    return result
