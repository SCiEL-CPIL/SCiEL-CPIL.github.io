import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Check for non-finite inputs
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return np.zeros_like(x)
        
    # Deadline guard: if we've already spent too much time, return a simple prediction
    # This is a safeguard, but the function is called repeatedly by the optimizer.
    # We assume the optimizer handles the timing, but we must be fast.
    
    # Structural Change: Radial Basis Function (RBF) / Gaussian Kernel approach
    # The Fourier basis assumes periodicity. The RBF basis assumes local smoothness.
    # We use 5 Gaussian centers. Each center (cx, cy) requires 3 params: weight, cx, cy.
    # Total params: 5 * 3 = 15. But we only have 10.
    # Alternative: Fixed centers, 5 weights + 5 width params? Or fixed centers, fixed width, 5 weights + 5 other?
    # Let's try: 5 fixed centers, 5 weights. That's only 5 params.
    # We have 10 params.
    # Let's use 5 centers.
    # Center 1: (0, 0)
    # Center 2: (0.5, 0)
    # Center 3: (-0.5, 0)
    # Center 4: (0, 0.5)
    # Center 5: (0, -0.5)
    # Params 0-4: Weights for these centers.
    # Params 5-9: What to do with them?
    # Maybe a linear trend: p5*x + p6*y + p7 + p8*x*y + p9*(x^2+y^2)?
    # This combines local bumps with global structure.
    
    # Centers
    cx = np.array([0.0, 0.5, -0.5, 0.0, 0.0])
    cy = np.array([0.0, 0.0, 0.0, 0.5, -0.5])
    
    # Compute distances from each point (x, y) to each center
    # x, y are 1D arrays of length N
    # We want a matrix of shape (N, 5)
    
    # dx = x[:, None] - cx[None, :]
    # dy = y[:, None] - cy[None, :]
    # dist_sq = dx**2 + dy**2
    
    # RBF term: exp(-dist_sq / (2 * sigma^2))
    # Let sigma be fixed, e.g., 0.5.
    sigma = 0.5
    dist_sq = (x[:, None] - cx[None, :])**2 + (y[:, None] - cy[None, :])**2
    rbf = np.exp(-dist_sq / (2 * sigma**2))
    
    # Prediction from RBF: sum(w_i * rbf_i)
    # params[0:5] are weights
    rbf_pred = rbf @ params[0:5]
    
    # Polynomial trend for the remaining 5 params
    # p5: constant
    # p6: x
    # p7: y
    # p8: x*y
    # p9: x^2 + y^2
    
    poly_pred = (params[5] * 1.0 + 
                 params[6] * x + 
                 params[7] * y + 
                 params[8] * (x * y) + 
                 params[9] * (x**2 + y**2))
    
    return rbf_pred + poly_pred
