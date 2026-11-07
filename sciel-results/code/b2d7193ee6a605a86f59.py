import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use RBF (Radial Basis Functions) with fixed centers spread across [-2, 2]
    # This provides a different functional family than Chebyshev polynomials
    # Centers are fixed at 10 evenly spaced points to avoid symmetry issues
    
    # Fixed centers spread across the domain [-2, 2]
    centers = np.array([-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    # Use 10 centers for 10 parameters
    centers = centers[:10]
    
    # Compute distances from x to each center
    # x has shape (n,), centers has shape (10,)
    # distances will have shape (n, 10)
    distances = np.abs(x[:, np.newaxis] - centers[np.newaxis, :])
    
    # Gaussian RBF kernel: exp(-beta * d^2)
    # Use a moderate width parameter
    beta = 2.0
    
    # Compute RBF values
    rbf_values = np.exp(-beta * distances ** 2)
    
    # prediction = params @ rbf_values.T
    # params has shape (10,), rbf_values has shape (n, 10)
    pred = rbf_values @ params
    
    return pred
