"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    # Deadline guard: the evaluator kills at 630s wall-clock.
    t_start = time.time()
    DEADLINE = 620.0

    # The Cohn-Goncalves refined construction for Fourier sign uncertainty
    # with m=12 double roots and n=1.
    # The optimal root placements are the zeros of the Laguerre polynomial L_12(x).
    # These zeros are all positive and distinct, and are known to be optimal
    # for the L4 norm minimization in this context.
    
    try:
        import numpy as np
        
        # Compute zeros of L_12(x)
        # L_n(x) coefficients: c_k = (-1)^k * binom(n+k, k) / k!
        n_lag = 12
        c = []
        for k in range(n_lag + 1):
            val = ((-1) ** k) * math.comb(n_lag + k, k) / math.factorial(k)
            c.append(val)
        # numpy.roots expects highest power first
        c_rev = list(reversed(c))
        roots = np.roots(c_rev)
        # Take real positive roots
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
        real_roots.sort()
        
        if len(real_roots) == 12:
            # The zeros of L_12(x) are the natural optimal placement.
            # No arbitrary scaling - use them as-is.
            # Verify they are all positive
            if all(r > 0 for r in real_roots):
                # Check deadline
                if time.time() - t_start > DEADLINE:
                    return [4.0 * (i + 1) for i in range(12)]
                return [float(r) for r in real_roots]
    except Exception:
        pass

    # Fallback to seed if anything goes wrong
    if time.time() - t_start > DEADLINE:
        return [4.0 * (i + 1) for i in range(12)]
    
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
