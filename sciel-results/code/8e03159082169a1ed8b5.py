import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Structural change: Use a product of two low-degree polynomials
    # P(x) = (a0 + a1*x + a2*x^2) * (b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4)
    # This gives a degree 6 polynomial with 10 parameters (3 + 5 = 8, but we have 10)
    # Actually, let me use: P(x) = (a0 + a1*x + a2*x^2 + a3*x^3) * (b0 + b1*x + b2*x^2 + b3*x^3)
    # That's 4 + 4 = 8 parameters. We have 10.
    # Let me use: P(x) = (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) * (b0 + b1*x + b2*x^2)
    # That's 5 + 3 = 8 parameters. Still need 2 more.
    # Let me use: P(x) = (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) * (b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4)
    # That's 5 + 5 = 10 parameters. This is a degree 9 polynomial, but factored.
    
    # However, this is mathematically equivalent to a single degree 9 polynomial,
    # so it doesn't add representational power. The benefit is that the optimization
    # landscape might be different.
    
    # Let me try a different structural change: use a sum of Gaussian-like bumps
    # centered at different locations, which can capture local features.
    # y(x) = sum_{i=0}^{4} w_i * exp(-((x - c_i)^2) / (2*s_i^2))
    # That's 5 weights + 5 centers + 5 scales = 15 parameters. Too many.
    # Let me use fixed scales: s_i = 1.0 for all i
    # y(x) = sum_{i=0}^{4} w_i * exp(-((x - c_i)^2) / 2)
    # That's 5 weights + 5 centers = 10 parameters.
    
    w = p[0:5]
    c = p[5:10]
    
    # Compute Gaussian bumps
    y = np.zeros_like(x)
    for i in range(5):
        y += w[i] * np.exp(-0.5 * (x - c[i])**2)
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
