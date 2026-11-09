import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params shape: (10,)
    # Split params:
    # 5 for polynomial coefficients: a0, a1, a2, a3, a4
    # 5 for Gaussian weights: w1, w2, w3, w4, w5
    
    a = params[:5]
    w = params[5:10]
    
    # Polynomial part: 1, x, x^2, x^3, x^4
    poly = np.empty_like(x)
    poly[:] = 1.0
    for i in range(1, 5):
        poly = poly * x
        # We need to accumulate the weighted sum, so we do it step by step
    # Let's just compute the polynomial sum directly
    p = np.zeros_like(x)
    for i in range(5):
        p = p + a[i] * (x ** i)
        
    # Gaussian part
    # Fixed centers
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0
    g = np.zeros_like(x)
    
    # Compute for each center
    for i in range(5):
        diff = x - centers[i]
        g = g + w[i] * np.exp(-0.5 * (diff / sigma) ** 2)
        
    return p + g
