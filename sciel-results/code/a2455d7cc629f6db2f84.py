# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of 10 diverse, scaled basis functions.
    
    The model is:
    f(x) = w0*1 + w1*(x/3) + w2*(x/3)^2 + w3*(x/3)^3 + w4*(x/3)^4 + w5*(x/3)^5
           + w6*sin(x) + w7*cos(x) + w8*tanh(x) + w9*exp(-x^2/2)
    
    Scaling the polynomial terms by (1/3)^k ensures that all basis functions
    are of order O(1) for x in [-3, 3]. This improves the condition number
    of the design matrix, helping BFGS converge more reliably and accurately.
    
    The basis includes:
    - Scaled polynomials up to degree 5.
    - Sine and cosine (frequency 1).
    - Tanh for saturation behavior.
    - A Gaussian bump for localized features.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Guard against non-finite inputs
    x = np.where(np.isfinite(x), x, 0.0)
    
    # Extract parameters (weights)
    w = params[:10]
    w = np.where(np.isfinite(w), w, 0.0)
    
    # Build the basis matrix Phi of shape (N, 10)
    
    # 1. Constant
    phi0 = np.ones_like(x)
    
    # 2-6. Scaled Polynomials
    # Normalizing by 3^k keeps values in [-1, 1] for x in [-3, 3]
    x1 = x / 3.0
    x2 = x1 * x1
    x3 = x2 * x1
    x4 = x2 * x2
    x5 = x4 * x1
    
    phi1 = x1
    phi2 = x2
    phi3 = x3
    phi4 = x4
    phi5 = x5
    
    # 7-8. Trigonometric
    phi6 = np.sin(x)
    phi7 = np.cos(x)
    
    # 9. Hyperbolic Tangent
    phi8 = np.tanh(x)
    
    # 10. Gaussian bump
    phi9 = np.exp(-0.5 * x**2)
    
    # Stack into matrix
    Phi = np.column_stack((phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9))
    
    # Compute prediction: y = Phi @ w
    result = Phi @ w
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
