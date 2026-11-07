import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    e = strain
    t = temp
    p = params
    
    # Basis 1-4: Polynomial/Interaction terms
    b1 = np.ones_like(e)
    b2 = e
    b3 = t
    b4 = e * t
    
    # Basis 5-10: Hinge functions (ReLU) at specific strain thresholds
    # These capture the non-linear hardening and yield behavior
    # Thresholds chosen to cover elastic, yield, and plastic regions
    c1 = 0.002
    c2 = 0.01
    c3 = 0.05
    c4 = 0.1
    c5 = 0.3
    c6 = 0.5
    
    b5 = np.maximum(0, e - c1)
    b6 = np.maximum(0, e - c2)
    b7 = np.maximum(0, e - c3)
    b8 = np.maximum(0, e - c4)
    b9 = np.maximum(0, e - c5)
    b10 = np.maximum(0, e - c6)
    
    sigma = (p[0] * b1 + 
             p[1] * b2 + 
             p[2] * b3 + 
             p[3] * b4 + 
             p[4] * b5 + 
             p[5] * b6 + 
             p[6] * b7 + 
             p[7] * b8 + 
             p[8] * b9 + 
             p[9] * b10)
             
    return sigma
