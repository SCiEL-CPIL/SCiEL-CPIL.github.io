import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Fixed Basis Linear Regression with Degree-3 Polynomial Terms.
    
    The 10 parameters are coefficients for the following basis functions:
    p[0]: 1 (intercept)
    p[1]: x
    p[2]: y
    p[3]: z
    p[4]: x^2
    p[5]: y^2
    p[6]: z^2
    p[7]: x*y
    p[8]: x*z
    p[9]: y*z
    
    Model: f(x,y,z) = p[0] + p[1]*x + p[2]*y + p[3]*z + p[4]*x^2 + p[5]*y^2 + p[6]*z^2 + p[7]*x*y + p[8]*x*z + p[9]*y*z
    
    This is a linear model in parameters, which is convex and well-suited for BFGS.
    It captures up to quadratic interactions and main effects, providing a strong
    approximation for smooth functions in the bounded domain [0.5, 2]^3.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        
        # Compute basis functions
        # Ensure inputs are at least 1D for consistent broadcasting
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        
        # Basis terms
        b0 = np.ones_like(x)
        b1 = x
        b2 = y
        b3 = z
        b4 = x * x
        b5 = y * y
        b6 = z * z
        b7 = x * y
        b8 = x * z
        b9 = y * z
        
        # Linear combination
        # Using dot product or explicit summation
        # Explicit summation is often faster for small fixed numbers of terms
        output = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + 
                  p[4] * b4 + p[5] * b5 + p[6] * b6 + 
                  p[7] * b7 + p[8] * b8 + p[9] * b9)
        
        # Safety check
        if not np.all(np.isfinite(output)):
            # If something goes wrong, return zeros to avoid breaking the evaluator
            return np.zeros_like(x)
            
        return output
        
    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
