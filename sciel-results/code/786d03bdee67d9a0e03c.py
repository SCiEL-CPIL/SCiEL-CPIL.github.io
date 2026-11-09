import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Centers for RBF
    # 1. (-1, -1)
    # 2. (1, -1)
    # 3. (-1, 1)
    # 4. (1, 1)
    # 5. (0, 0)
    
    # Calculate squared distances to each center
    # d1^2 = (x - cx1)^2 + (y - cy1)^2
    
    # Center 1: (-1, -1)
    d1_sq = (x + 1.0)**2 + (y + 1.0)**2
    # Center 2: (1, -1)
    d2_sq = (x - 1.0)**2 + (y + 1.0)**2
    # Center 3: (-1, 1)
    d3_sq = (x + 1.0)**2 + (y - 1.0)**2
    # Center 4: (1, 1)
    d4_sq = (x - 1.0)**2 + (y - 1.0)**2
    # Center 5: (0, 0)
    d5_sq = x**2 + y**2
    
    # Parameters p[0:5] are amplitudes
    # Parameters p[5:10] are log-widths (gamma = exp(p))
    # Using exp ensures gamma > 0, which is required for decaying Gaussian kernels.
    
    c1 = p[0]
    c2 = p[1]
    c3 = p[2]
    c4 = p[3]
    c5 = p[4]
    
    g1 = np.exp(p[5])
    g2 = np.exp(p[6])
    g3 = np.exp(p[7])
    g4 = np.exp(p[8])
    g5 = np.exp(p[9])
    
    # Compute RBF terms
    # Clipping the exponent to prevent overflow/underflow issues during optimization
    # exp(-g * d_sq)
    # If g is large and d_sq is large, exponent is very negative -> 0.
    # If g is small, exponent is close to 0 -> 1.
    # If g is huge, and d_sq is 0, exponent is 0 -> 1.
    # We should be careful with numerical stability.
    
    term1 = c1 * np.exp(-g1 * d1_sq)
    term2 = c2 * np.exp(-g2 * d2_sq)
    term3 = c3 * np.exp(-g3 * d3_sq)
    term4 = c4 * np.exp(-g4 * d4_sq)
    term5 = c5 * np.exp(-g5 * d5_sq)
    
    return term1 + term2 + term3 + term4 + term5
