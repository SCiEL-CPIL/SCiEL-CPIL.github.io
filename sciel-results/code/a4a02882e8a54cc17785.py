import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s limit

    try:
        m = 12
        # The directive suggests using L_12^(1) for dimension n=1
        # Standard Laguerre is L_12^(0). The seed (score 1.0) likely used L_12^(0).
        # We test if L_12^(1) provides a tighter bound (score > 1.0).
        
        # Get roots of generalized Laguerre polynomial L_12^(1)
        # alpha = 1
        roots, _ = roots_genlaguerre(m, 1.0)
        
        # Ensure roots are positive (they should be for alpha > -1)
        if not np.all(roots > 0):
            # Fallback to standard if something went wrong (shouldn't happen)
            roots, _ = roots_genlaguerre(m, 0.0)
            
        # Sort for consistency
        r = sorted(roots)
        
        # Check time (though this operation is very fast)
        if time.time() > deadline:
             # Fallback to seed if somehow slow (unlikely)
             return [4.0 * (i + 1) for i in range(12)]
             
        return [float(x) for x in r]
        
    except Exception:
        # Fallback to standard Laguerre roots (the seed)
        try:
            roots, _ = roots_genlaguerre(12, 0.0)
            return [float(x) for x in sorted(roots)]
        except Exception:
            # Last resort fallback
            return [4.0 * (i + 1) for i in range(12)]
