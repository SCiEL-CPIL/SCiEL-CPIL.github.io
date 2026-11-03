import numpy as np
import time

def construct():
    """
    Constructs a signed step function of length 400 using Legendre symbols
    modulo a prime p > 400 to minimize the peak aperiodic autocorrelation.
    """
    start_time = time.time()
    deadline = 120.0  # 120s limit
    
    N = 400
    
    # Select a prime p > N
    # 401 is the smallest prime greater than 400
    p = 401
    
    # Generate the sequence
    # s_i = Legendre(i, p) for i = 1, ..., N
    # Legendre symbol is 1 if i is a quadratic residue mod p, -1 otherwise.
    # Since i < p, i is never 0 mod p.
    
    seq = np.empty(N, dtype=np.int8)
    
    # Optimization: Precompute powers or use efficient pow
    # Euler's criterion: (a/p) = a^((p-1)/2) mod p
    exp = (p - 1) // 2
    
    for i in range(N):
        val = i + 1
        # Calculate Legendre symbol
        # pow(base, exp, mod) is fast in Python
        ls = pow(val, exp, p)
        if ls == 1:
            seq[i] = 1
        else:
            seq[i] = -1
            
    # Simple sanity check / deadline guard
    # The computation is very fast, but we respect the directive.
    if time.time() - start_time > deadline - 5.0:
        # Return current best (which is the full sequence)
        pass
        
    return seq
