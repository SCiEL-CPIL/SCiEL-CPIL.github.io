import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs for stability
    # b: [1, 5] -> [-1, 1]
    b_norm = (b - 3.0) / 2.0
    
    # s: [1, 40] -> [0, 1] approx
    s_norm = (s - 1.0) / 39.0
    
    # temp: [1, 50] -> [-1, 1]
    t_norm = (temp - 25.5) / 24.5
    
    # pH: [4, 10] -> [-1, 1]
    ph_norm = (pH - 7.0) / 3.0
    
    # Stable sigmoid function to avoid overflow
    def sigmoid(x):
        return np.where(x >= 0, 
                        1.0 / (1.0 + np.exp(-x)), 
                        np.exp(x) / (1.0 + np.exp(x)))
    
    # Approach: Sum of two Sigmoidal Planes
    # This is a distinct approach from the multiplicative Monod/Gaussian model.
    # It allows for additive effects and sharp transitions.
    
    # First Plane: Amplitude p0, Weights p1-p4
    x1 = p1 * b_norm + p2 * s_norm + p3 * t_norm + p4 * ph_norm
    term1 = p0 * sigmoid(x1)
    
    # Second Plane: Amplitude p5, Weights p6-p9
    x2 = p6 * b_norm + p7 * s_norm + p8 * t_norm + p9 * ph_norm
    term2 = p5 * sigmoid(x2)
    
    return term1 + term2
