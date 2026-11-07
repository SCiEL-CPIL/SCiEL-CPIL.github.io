import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    if not np.all(np.isfinite(p)):
        p = np.ones(10)
        
    bias = p[0]
    
    amps = p[1:4]
    centers = 2.0 * np.tanh(p[4:7])
    widths = 0.5 * np.exp(p[7:10])
    
    # Compute sum of Gaussians
    # x is (N,), centers is (3,), widths is (3,)
    # Need broadcasting: x[:, None] - centers[None, :]
    x_expanded = x[:, np.newaxis]
    diff = x_expanded - centers[np.newaxis, :]
    
    # exp(-0.5 * (diff / widths)^2)
    # widths is (3,), so diff/widths is (N, 3)
    gaussians = np.exp(-0.5 * (diff / widths[np.newaxis, :])**2)
    
    # Weighted sum
    # gaussians is (N, 3), amps is (3,)
    bumps = np.sum(gaussians * amps[np.newaxis, :], axis=1)
    
    return bias + bumps
