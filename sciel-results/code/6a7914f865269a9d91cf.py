import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    start_time = time.time()
    deadline = start_time + 580.0
    
    try:
        m, n = 12, 1
        N_ROOTS = 12
        
        # For the Cohn-Goncalves construction with Fourier sign uncertainty,
        # the optimal double-root locations are 2 * roots of L_m^{(2n+1)}(x)
        # where L_m^{(alpha)} is the generalized Laguerre polynomial.
        # For m=12, n=1: alpha = 2*1+1 = 3
        
        alpha = 2 * n + 1  # = 3
        
        # Get roots of L_12^{(3)}(x)
        roots_laguerre, weights = roots_genlaguerre(m, alpha)
        
        # The double-root locations are 2 * these roots
        # (scaling accounts for the Gaussian e^{-x^2/4} vs standard Laguerre weight)
        double_roots = 2.0 * roots_laguerre
        
        # Sort and ensure positive
        double_roots = np.sort(double_roots)
        double_roots = np.maximum(double_roots, 1e-10)
        
        # Validate: must be 12 positive finite values
        if len(double_roots) != N_ROOTS:
            return [4.0 * (i + 1) for i in range(N_ROOTS)]
        if np.any(~np.isfinite(double_roots)) or np.any(double_roots <= 0):
            return [4.0 * (i + 1) for i in range(N_ROOTS)]
        
        return [float(r) for r in double_roots]
        
    except Exception:
        return [4.0 * (i + 1) for i in range(N_ROOTS)]

if __name__ == "__main__":
    construct()
