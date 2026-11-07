import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Hybrid basis function approach with a rich set of non-polynomial terms.
    
    Instead of a fixed neural network structure or a simple polynomial, we use a 
    carefully selected basis of 10 functions that covers:
    - Constant term
    - Linear terms (x, y, z)
    - Exponential terms (exp(x), exp(y), exp(z))
    - Logarithmic terms (log(x), log(y), log(z))
    
    This is a linear-in-parameters model: output = sum(params[i] * basis[i])
    
    The key insight is that if the target function contains exponential or 
    logarithmic components, a linear combination of these basis functions can 
    represent them exactly (up to numerical precision), whereas polynomials or 
    small neural networks with limited capacity would struggle.
    
    Since the model is linear in params, BFGS will converge quickly and reliably 
    to the optimal solution, avoiding local minima issues that plague nonlinear 
    parameterizations.
    """
    try:
        n = len(x)
        
        # Ensure inputs are 1D arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        
        # Clip inputs to avoid log(0) or extreme exp values
        # Inputs are from [0.5, 2], so this is safe, but add margin
        x_safe = np.clip(x, 1e-10, 100)
        y_safe = np.clip(y, 1e-10, 100)
        z_safe = np.clip(z, 1e-10, 100)
        
        # Build the 10 basis functions
        # p[0]: constant
        # p[1]: x
        # p[2]: y
        # p[3]: z
        # p[4]: exp(x)
        # p[5]: exp(y)
        # p[6]: exp(z)
        # p[7]: log(x)
        # p[8]: log(y)
        # p[9]: log(z)
        
        p = params
        
        # Compute basis terms
        b0 = np.ones_like(x)
        b1 = x
        b2 = y
        b3 = z
        b4 = np.exp(x_safe)
        b5 = np.exp(y_safe)
        b6 = np.exp(z_safe)
        b7 = np.log(x_safe)
        b8 = np.log(y_safe)
        b9 = np.log(z_safe)
        
        # Linear combination
        output = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
                  p[4] * b4 + p[5] * b5 + p[6] * b6 +
                  p[7] * b7 + p[8] * b8 + p[9] * b9)
        
        # Safety check: Ensure finite output
        if not np.all(np.isfinite(output)):
            # Fallback: return zeros if something went wrong
            return np.zeros(n)
            
        return output

    except Exception:
        # In case of any error, return zeros
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
