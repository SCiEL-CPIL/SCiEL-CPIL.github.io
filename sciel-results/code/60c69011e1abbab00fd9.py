import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The task specifies m=12 and n=1.
    # Directive: Use the exact zeros of L_12^(1)(x).
    # We use scipy.special.roots_genlaguerre for high-precision roots.
    # Degree n=12, parameter alpha=1.
    roots, _ = roots_genlaguerre(12, 1)
    
    # Convert to list of floats and sort to ensure consistent ordering
    return sorted([float(r) for r in roots])
