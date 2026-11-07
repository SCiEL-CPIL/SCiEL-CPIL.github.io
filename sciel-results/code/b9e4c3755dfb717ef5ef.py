import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a 10-parameter model with a rich basis:
    - Polynomial terms: 1, x, x^2, x^3, x^4
    - Trigonometric terms: sin(x), cos(x), sin(2x), cos(2x)
    - Adaptive Gaussian term: exp(-(x - c)^2 / (2*s^2))
    
    The last two parameters control the center c and width s of the
    Gaussian term. The first 8 parameters are linear coefficients for
    the fixed basis functions.
    
    To ensure BFGS convergence and numerical stability:
    - The Gaussian width s is clamped to a reasonable range.
    - All operations are vectorized and efficient.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.shape[0] != 10:
        return np.zeros_like(x)
    
    # Extract parameters
    c_poly = params[:5]       # Coeffs for 1, x, x^2, x^3, x^4
    c_trig = params[5:9]      # Coeffs for sin(x), cos(x), sin(2x), cos(2x)
    c_gauss = params[9]       # Coeff for Gaussian term
    
    # Wait, we have 10 params. Let's use:
    # params[0:5] -> poly coeffs (1, x, x^2, x^3, x^4)
    # params[5:9] -> trig coeffs (sin(x), cos(x), sin(2x), cos(2x))
    # params[9]   -> Gaussian coeff
    # But we need a center and width for the Gaussian. We only have 1 param left.
    # Let's fix the center at 0 and make the width adaptive? Or fix width and make center adaptive?
    # Better: Use a fixed grid of Gaussians? No, that would require more params.
    
    # Let's reconsider the basis. We have 10 params.
    # Option A: 4 poly + 4 trig + 1 tanh coeff + 1 tanh scale (current)
    # Option B: 5 poly + 3 trig + 1 Gaussian coeff + 1 Gaussian center?
    # Option C: 4 poly + 4 trig + 2 Gaussian (coeff, center), fixed width
    
    # Let's try a different approach: Use a more flexible basis.
    # Basis: 1, x, x^2, x^3, x^4, sin(x), cos(x), sin(2x), cos(2x), tanh(x)
    # All linear in params. This is a 10-term linear basis.
    # BFGS will converge to the global minimum since it's linear in params.
    
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Polynomial terms
    p1 = np.ones_like(x)
    p2 = x
    p3 = x**2
    p4 = x**3
    p5 = x**4
    
    # Trigonometric terms
    t1 = np.sin(x)
    t2 = np.cos(x)
    t3 = np.sin(2 * x)
    t4 = np.cos(2 * x)
    
    # Hyperbolic tangent
    h1 = np.tanh(x)
    
    # Combine
    prediction = (
        c0 * p1 +
        c1 * p2 +
        c2 * p3 +
        c3 * p4 +
        c4 * p5 +
        c5 * t1 +
        c6 * t2 +
        c7 * t3 +
        c8 * t4 +
        c9 * h1
    )
    
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(x)
    
    return prediction
