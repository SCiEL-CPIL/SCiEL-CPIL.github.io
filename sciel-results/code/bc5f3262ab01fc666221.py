"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import fsolve


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x)=0 using the Levenberg-Marquardt algorithm
    (via scipy.optimize.fsolve). This is a hybrid Newton method that is
    generally faster and more accurate for smooth systems than derivative-free
    methods like Nelder-Mead.
    """
    x0 = np.array(x0, dtype=float)
    
    def func(x):
        try:
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,):
                fx = fx.reshape(2)
            if not np.all(np.isfinite(fx)):
                return np.array([1e10, 1e10])
            return fx
        except Exception:
            return np.array([1e10, 1e10])

    try:
        # maxfev=450 to stay well under the 500 call limit
        sol, info, ier, msg = fsolve(func, x0, maxfev=450, full_output=True)
        
        # Check if solution is finite
        if not np.all(np.isfinite(sol)):
            return x0
            
        return sol
    except Exception:
        return x0
# EVOLVE-BLOCK-END
