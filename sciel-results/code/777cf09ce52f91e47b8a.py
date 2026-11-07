import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    e = strain
    t = temp
    p = params
    
    # Avoid division by zero or negative sqrt
    # E, H, K should be positive. BFGS might go negative.
    # We can clip them or use softplus? 
    # Clipping is not differentiable at 0. 
    # However, if BFGS goes negative, the model breaks.
    # Let's assume the optimizer stays in reasonable bounds or we add a small epsilon.
    
    E = p[0] + p[1]*t + p[7]*t**2
    H = p[2] + p[3]*t + p[8]*t**2
    K = p[4] + p[5]*t + p[9]*t**2
    
    # Safety: ensure positive denominators
    E = np.maximum(E, 1e-6)
    H = np.maximum(H, 1e-6)
    K = np.maximum(K, 1e-6)
    
    A = 1.0 / (K**2)
    B = 1.0 / E + 1.0 / H
    
    # Stable quadratic solution for sigma
    # sigma = 2 * e / (B + sqrt(B^2 + 4 * A * e))
    # Note: if e is negative (compression?), strain range is [-0.0019, 0.96].
    # The formula works for negative e?
    # B^2 + 4 A e. If e is negative, term inside sqrt might be < B^2.
    # sqrt is real as long as B^2 + 4 A e >= 0.
    # B is positive. A is positive.
    # If e is very negative, it might fail.
    # But strain is mostly positive.
    
    disc = B**2 + 4.0 * A * e
    # Ensure disc is non-negative for sqrt
    disc = np.maximum(disc, 0.0)
    
    sigma = (2.0 * e) / (B + np.sqrt(disc))
    
    return sigma + p[6]
