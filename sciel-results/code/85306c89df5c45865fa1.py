import numpy as np
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    # We use a 3-point distribution: {0, 1, M}
    # Weights w0, w1, w2 sum to 1.
    # We optimize w1 and w2. w0 = 1 - w1 - w2.
    
    def objective(params):
        w1, w2 = params
        w0 = 1.0 - w1 - w2
        if w0 < 0 or w1 < 0 or w2 < 0:
            return 100.0 # Penalty
        
        # Calculate P for support {0, 1, M} with M large
        # Using the derived formula for M >> 1
        # Term 1: w0*w1*(w0^2 + 2*w0*w1)
        t1 = w0 * w1 * (w0**2 + 2 * w0 * w1)
        # Term 2: w1^2 * w0^2
        t2 = w1**2 * w0**2
        # Term 3: (w0+w1)*w2*(1-w2^2)
        t3 = (w0 + w1) * w2 * (1 - w2**2)
        # Term 4: w2^2 * (1-w2)^2
        t4 = w2**2 * (1 - w2)**2
        
        P = t1 + t2 + t3 + t4
        return -P # Minimize negative P

    # Initial guess
    x0 = [0.2, 0.2]
    bounds = [(1e-6, 1-1e-6), (1e-6, 1-1e-6)]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    w1, w2 = res.x
    w0 = 1.0 - w1 - w2
    
    # Ensure non-negative
    if w0 < 0: w0 = 0
    if w1 < 0: w1 = 0
    if w2 < 0: w2 = 0
    
    # Normalize
    s = w0 + w1 + w2
    w0 /= s
    w1 /= s
    w2 /= s
    
    support = np.array([0.0, 1.0, 1000.0], dtype=np.float64)
    weights = np.array([w0, w1, w2], dtype=np.float64)
    
    return support, weights
