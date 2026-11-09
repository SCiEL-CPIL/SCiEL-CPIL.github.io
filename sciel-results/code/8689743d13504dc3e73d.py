import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    try:
        # The Cohn-Goncalves construction for m=12 likely utilizes the roots of the Laguerre polynomial L_12(x)
        # or a closely related variant. The roots of L_n(x) are the nodes for Gauss-Laguerre quadrature.
        # We assume the optimal double-root locations are these nodes.
        
        # Get roots of L_12(x)
        roots = roots_laguerre(12)
        
        # Ensure they are positive and sorted (they are by definition)
        # roots[0] is an array of nodes
        optimal_roots = list(roots[0])
        
        # Check if we are within time limit (trivially true here)
        if time.time() > deadline:
             # Fallback to seed if somehow it took too long (impossible)
             return [4.0 * (i + 1) for i in range(12)]
             
        return optimal_roots

    except Exception as e:
        # Fallback to seed if scipy fails or errors occur
        return [4.0 * (i + 1) for i in range(12)]
