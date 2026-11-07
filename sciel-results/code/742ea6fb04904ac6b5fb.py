import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a rational basis with a quadratic 
    numerator and a quadratic denominator. This structure allows the model 
    to capture saturation, poles, and non-polynomial smooth variations 
    that a fixed polynomial basis cannot represent efficiently.
    
    The model is:
    f(x, y) = (p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy) 
               / (1 + p6*x^2 + p7*y^2 + p8*xy + p9*(x^2 + y^2))
    
    Parameters:
    p0..p5: Numerator coefficients (constant, linear, quadratic, cross)
    p6..p9: Denominator coefficients (quadratic terms, ensuring stability 
            with a constant 1 offset)
            
    This non-linear parameterization provides a richer hypothesis class 
    than a linear combination of polynomials, potentially matching 
    rational or smoothly saturating target functions.
    """
    try:
        # Unpack parameters
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
        
        # Precompute powers
        x2 = x * x
        y2 = y * y
        xy = x * y
        x2_y2 = x2 + y2
        
        # Numerator
        num = p0 + p1 * x + p2 * y + p3 * x2 + p4 * y2 + p5 * xy
        
        # Denominator
        # We add 1 to ensure the denominator is non-zero when params are small or zero,
        # and to keep the scale reasonable when params are initialized to 1.
        den = 1.0 + p6 * x2 + p7 * y2 + p8 * xy + p9 * x2_y2
        
        # Avoid division by zero
        # If den is close to zero, clip it to prevent inf/nan
        # Since x,y in [-1,1] and params start at 1, den is likely safe, 
        # but during optimization, params can go negative.
        # We use a small epsilon to maintain gradient flow if possible, 
        # or just clip to a safe bound.
        den_safe = np.where(np.abs(den) < 1e-6, 1e-6, den)
        # Preserve sign if den was negative? 
        # Usually, for MSE, sign flip in denominator just flips sign of output, 
        # which can be compensated by numerator signs. 
        # However, to be safe against discontinuities, we can just use the safe version.
        
        prediction = num / den_safe
        
        if not np.all(np.isfinite(prediction)):
            return np.zeros_like(x)
            
        return prediction
        
    except Exception:
        return np.zeros_like(x)
