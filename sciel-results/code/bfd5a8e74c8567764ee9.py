import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Map x from [-2, 2] to theta in [-pi, pi]
    # This allows us to use standard Fourier frequencies k=1, 2, ...
    theta = x * (np.pi / 2.0)
    
    # Construct the Fourier basis matrix
    # We have 10 parameters.
    # Strategy: 
    # p[0]: DC offset (1)
    # p[1], p[2]: cos(theta), sin(theta)
    # p[3], p[4]: cos(2*theta), sin(2*theta)
    # p[5], p[6]: cos(3*theta), sin(3*theta)
    # p[7], p[8]: cos(4*theta), sin(4*theta)
    # p[9]: cos(5*theta)
    
    # Pre-compute trigonometric values for efficiency and stability
    cos1 = np.cos(theta)
    sin1 = np.sin(theta)
    cos2 = np.cos(2.0 * theta)
    sin2 = np.sin(2.0 * theta)
    cos3 = np.cos(3.0 * theta)
    sin3 = np.sin(3.0 * theta)
    cos4 = np.cos(4.0 * theta)
    sin4 = np.sin(4.0 * theta)
    cos5 = np.cos(5.0 * theta)
    
    # Build the linear combination
    # Note: This is a linear model y = X @ p
    # This is ideal for BFGS as the Hessian is constant (2 * X^T X)
    
    y = np.ones_like(x) * p[0]
    y += p[1] * cos1
    y += p[2] * sin1
    y += p[3] * cos2
    y += p[4] * sin2
    y += p[5] * cos3
    y += p[6] * sin3
    y += p[7] * cos4
    y += p[8] * sin4
    y += p[9] * cos5
    
    return y
