import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Fixed-Basis Linear Regression with Quadratic + Interaction Terms.
    
    Instead of optimizing non-convex RBF centers/widths, we use a fixed, 
    deterministic basis of 10 terms. The 10 parameters are simply the linear 
    coefficients for this basis. This turns the fitting problem into a convex 
    quadratic optimization, which BFGS solves reliably and quickly.
    
    Basis Terms:
    1. 1 (Intercept)
    2. x
    3. y
    4. z
    5. x^2
    6. y^2
    7. z^2
    8. x*y
    9. x*z
    10. y*z
    
    This basis can represent any quadratic surface and some interaction effects,
    which is often sufficient for smooth functions on [0.5, 2]^3.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        
        # Ensure p has at least 10 elements, pad with zeros if necessary
        if len(p) < 10:
            p = np.pad(p, (0, 10 - len(p)), constant_values=0)
        else:
            p = p[:10]
            
        # Compute basis features
        # We compute these efficiently using broadcasting
        # n = len(x)
        
        # 1. Intercept
        b1 = np.ones_like(x)
        
        # 2-4. Linear terms
        b2 = x
        b3 = y
        b4 = z
        
        # 5-7. Squared terms
        b5 = x * x
        b6 = y * y
        b7 = z * z
        
        # 8-10. Interaction terms
        b8 = x * y
        b9 = x * z
        b10 = y * z
        
        # Combine linear combination
        # output = p[0]*b1 + p[1]*b2 + ... + p[9]*b10
        
        output = (
            p[0] * b1 +
            p[1] * b2 +
            p[2] * b3 +
            p[3] * b4 +
            p[4] * b5 +
            p[5] * b6 +
            p[6] * b7 +
            p[7] * b8 +
            p[8] * b9 +
            p[9] * b10
        )
        
        # Safety check for non-finite values
        if not np.all(np.isfinite(output)):
            return np.zeros_like(x)
            
        return output
        
    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
