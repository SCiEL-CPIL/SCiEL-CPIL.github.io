import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    # RBF Centers
    centers = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ])
    
    # Fixed width parameter for Gaussian RBFs
    # Tuned for domain [-1, 1]
    gamma = 2.0 
    
    # Calculate RBF terms
    # x, y are arrays of shape (N,)
    rbf_terms = np.zeros((5, len(x)))
    for i in range(5):
        cx, cy = centers[i]
        dist_sq = (x - cx)**2 + (y - cy)**2
        rbf_terms[i] = np.exp(-gamma * dist_sq)
    
    # Polynomial background terms
    # 1, x, y, x^2, y^2
    poly_terms = np.zeros((5, len(x)))
    poly_terms[0] = 1.0
    poly_terms[1] = x
    poly_terms[2] = y
    poly_terms[3] = x * x
    poly_terms[4] = y * y
    
    # Combine
    # Params 0-4 for RBFs, 5-9 for Poly
    rbf_weights = p[0:5]
    poly_weights = p[5:10]
    
    pred = np.sum(rbf_weights[:, None] * rbf_terms, axis=0) + np.sum(poly_weights[:, None] * poly_terms, axis=0)
    
    return pred
