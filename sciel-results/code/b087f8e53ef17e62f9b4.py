"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    """
    t0 = time.time()
    deadline = t0 + 570
    
    m = 12
    n = 1
    
    try:
        from scipy.special import roots_laguerre
        nodes, _ = roots_laguerre(m, alpha=0)
        result = [float(x) for x in nodes]
        if len(result) == m and all(x > 0 for x in result):
            return result
    except:
        pass
    
    try:
        nodes, _ = np.polynomial.laguerre.laggauss(m)
        result = [float(x) for x in nodes]
        if len(result) == m and all(x > 0 for x in result):
            return result
    except:
        pass
    
    return [4.0 * (i + 1) for i in range(m)]
# EVOLVE-BLOCK-END
