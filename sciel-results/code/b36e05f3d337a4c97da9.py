import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # Strategy: Rational Function Approximation
    # y(x) = N(x) / D(x)
    # N(x) is a polynomial of degree 4 (5 coefficients: p[0] to p[4])
    # D(x) is a polynomial of degree 4 (5 coefficients: p[5] to p[9])
    #
    # This creates a non-convex optimization landscape, but the structure 
    # allows for fitting functions with more complex local behaviors 
    # (e.g., peaks, valleys, or asymptotic behaviors) than a fixed linear 
    # combination of basis functions, provided BFGS can navigate the 
    # parameter space.
    #
    # To avoid division by zero or near-zero denominators which cause 
    # instability and large gradients, we add a small epsilon to the 
    # denominator. We also clip the denominator to prevent extreme values.
    
    # Numerator coefficients
    n0, n1, n2, n3, n4 = p[0], p[1], p[2], p[3], p[4]
    # Denominator coefficients
    d0, d1, d2, d3, d4 = p[5], p[6], p[7], p[8], p[9]
    
    # Compute polynomials
    # Using Horner's method for efficiency and numerical stability
    # N(x) = n4*x^4 + n3*x^3 + n2*x^2 + n1*x + n0
    # D(x) = d4*x^4 + d3*x^3 + d2*x^2 + d1*x + d0
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    numerator = n4 * x4 + n3 * x3 + n2 * x2 + n1 * x + n0
    denominator = d4 * x4 + d3 * x3 + d2 * x2 + d1 * x + d0
    
    # Stabilize denominator
    # If the denominator is close to zero, the gradient can be huge.
    # We replace small values with a small constant to keep the function 
    # finite and differentiable, effectively acting as a regularizer.
    # The sign is preserved to maintain continuity where possible.
    eps = 1e-6
    sign_d = np.sign(denominator)
    # Where denominator is 0, sign is 0. We want to avoid 0/0.
    # If d is 0, we want to replace it with eps (or -eps if we knew direction, 
    # but 0 is neutral). Let's just use np.where to handle the magnitude.
    
    # A more robust approach: 
    # denom_safe = np.where(np.abs(denominator) < eps, eps, denominator)
    # But this introduces a discontinuity in the derivative at eps.
    # BFGS might struggle with this.
    
    # Alternative: Add a constant to the denominator to shift it away from zero?
    # No, that changes the function class.
    
    # Let's try a smooth regularization:
    # denom_safe = denominator + eps * np.sign(denominator) ? No, sign(0)=0.
    
    # Standard trick: denom_safe = np.where(np.abs(denominator) < eps, 
    #                                        eps * np.where(denominator >= 0, 1, -1), 
    #                                        denominator)
    # This is still kinked.
    
    # Given the risk of BFGS failing on non-smooth or ill-conditioned problems,
    # and the fact that the "linear basis" program scored ~4.8e9, 
    # a rational function might score worse if it fails to converge.
    # However, the directive asks for a structural change.
    
    # Let's use a very simple stabilization:
    denom_safe = np.where(np.abs(denominator) < 1e-4, 1e-4, denominator)
    
    y = numerator / denom_safe
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        # If anything blows up, return zeros (or the last finite value?)
        # Returning zeros is a safe fallback that yields a high but finite MSE.
        return np.zeros_like(x)
        
    return y
