"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid basis of radial basis functions (RBFs) 
    and low-order polynomials. This structure allows the model to capture localized 
    non-uniform smooth structures and complex multi-modal behaviors that global 
    polynomial or Fourier bases may miss, while maintaining numerical stability 
    for the BFGS optimizer.
    """
    start_time = time.time()
    
    try:
        # Ensure inputs are float arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Basis Terms
        # 1. Constant
        term0 = np.ones_like(x)
        
        # 2. Linear x
        term1 = x
        
        # 3. Linear y
        term2 = y
        
        # 4. Quadratic x
        term3 = x**2
        
        # 5. Quadratic y
        term4 = y**2
        
        # 6. Interaction xy
        term5 = x * y
        
        # Radial Basis Functions (Gaussian kernels)
        # Centers chosen to cover the domain [-1, 1] x [-1, 1]
        # Widths chosen to be moderately localized but overlapping
        
        # 7. RBF centered at (0, 0)
        r0 = np.sqrt(x**2 + y**2)
        term6 = np.exp(-2.0 * r0**2)
        
        # 8. RBF centered at (0.5, 0.5)
        r1 = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        term7 = np.exp(-2.0 * r1**2)
        
        # 9. RBF centered at (-0.5, -0.5)
        r2 = np.sqrt((x + 0.5)**2 + (y + 0.5)**2)
        term8 = np.exp(-2.0 * r2**2)
        
        # 10. RBF centered at (0.5, -0.5)
        r3 = np.sqrt((x - 0.5)**2 + (y + 0.5)**2)
        term9 = np.exp(-2.0 * r3**2)
        
        # Combine linearly
        prediction = (params[0] * term0 + 
                      params[1] * term1 + 
                      params[2] * term2 + 
                      params[3] * term3 + 
                      params[4] * term4 + 
                      params[5] * term5 + 
                      params[6] * term6 + 
                      params[7] * term7 + 
                      params[8] * term8 + 
                      params[9] * term9)
        
        # Check for non-finite values
        if not np.all(np.isfinite(prediction)):
             return np.zeros_like(x)
             
        return prediction
        
    except Exception:
        return np.zeros_like(x)
# EVOLVE-BLOCK-END
