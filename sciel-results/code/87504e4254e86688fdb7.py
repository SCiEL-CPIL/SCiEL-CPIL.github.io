"""Optimized product-formula coefficients for six-site Heisenberg chain splitting."""
import numpy as np
from numpy.linalg import norm
import time

# Deadline guard
_DEADLINE = 130.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def coefficients(stages):
    """
    Returns coefficients for a product formula approximating exp(-it(A+B)).
    Structure: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    Uses 11 substeps of 2nd-order Strang splitting, mapped to 12 pairs.
    """
    # 11 substeps of Strang
    # B coeffs: 1/22, 1/11, ..., 1/11, 1/22
    # A coeffs: 1/11, ..., 1/11, 0
    n_substeps = 11
    dt = 1.0 / n_substeps
    
    a = []
    b = []
    
    # Pair 1
    b.append(0.5 * dt)
    a.append(dt)
    
    # Pairs 2 to 11
    for _ in range(n_substeps - 1):
        b.append(dt)
        a.append(dt)
        
    # Pair 12
    b.append(0.5 * dt)
    a.append(0.0)
    
    # Ensure lengths are 12
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
        
    # Truncate if stages < 12 (though task says 12)
    a = a[:stages]
    b = b[:stages]
    
    return a, b
