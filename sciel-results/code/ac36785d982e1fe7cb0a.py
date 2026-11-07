import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Structural change: Mix of polynomial and RBF basis functions
    # This allows capturing both global trends and local oscillations/features
    
    # Polynomial basis (first 4 params)
    p0 = np.ones_like(x)
    p1 = x
    p2 = x**2
    p3 = x**3
    
    # RBF basis (next 6 params)
    # Fixed centers and widths
    centers = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    sigma = 1.0
    
    rbf_terms = np.zeros((len(x), 6))
    for i, c in enumerate(centers):
        rbf_terms[:, i] = np.exp(-(x - c)**2 / (2 * sigma**2))
        
    # Combine basis
    # We have 4 poly + 6 rbf = 10 basis functions
    # Map params[0:4] to poly, params[4:10] to rbf
    
    # Ensure x is 1D for safety in case of shape issues, though input is likely 1D
    x_flat = np.asarray(x).flatten()
    p0 = np.ones_like(x_flat)
    p1 = x_flat
    p2 = x_flat**2
    p3 = x_flat**3
    
    rbf_terms = np.zeros((len(x_flat), 6))
    for i, c in enumerate(centers):
        rbf_terms[:, i] = np.exp(-(x_flat - c)**2 / (2 * sigma**2))
        
    basis = np.column_stack((p0, p1, p2, p3, rbf_terms))
    
    # params should be length 10
    # If params length is different, pad or truncate? 
    # Contract says params is length 10.
    
    prediction = basis @ params
    
    # Reshape back to original input shape if needed, but return array is fine
    return prediction.reshape(np.asarray(x).shape)
