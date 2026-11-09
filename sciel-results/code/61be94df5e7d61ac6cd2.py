import numpy as np
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    # Use the roots of the generalized Laguerre polynomial L_12^(1)(x)
    # This corresponds to the roots of the derivative of L_13(x).
    # In many extremal problems involving derivatives or squared norms,
    # the alpha=1 case (associated with the derivative structure) 
    # provides tighter bounds than the standard alpha=0 case.
    roots, _ = roots_genlaguerre(12, 1)
    return [float(r) for r in roots]
# EVOLVE-BLOCK-END
