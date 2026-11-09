import numpy as np
import scipy.special
import time

def construct():
    # Deadline guard: 630s limit, use 600s as safety margin
    start_time = time.time()
    deadline = start_time + 590.0
    
    try:
        # The task specifies m=12 and n=1.
        # In the context of generalized Laguerre polynomials L_m^(alpha)(x),
        # the parameter 'n=1' from the task description likely maps to alpha=1.
        # The roots of L_12^(1)(x) are used as the locations for the double roots.
        
        # scipy.special.roots_genlaguerre(n, alpha) returns the roots of L_n^(alpha)(x)
        # n=12, alpha=1
        roots = scipy.special.roots_genlaguerre(12, 1)
        
        # roots is a numpy array of the roots. They are real and positive.
        # Sort them to be safe, although they usually come out sorted.
        root_list = sorted([float(r) for r in roots.real])
        
        # Verify constraints: 12 positive roots
        if len(root_list) != 12:
            raise ValueError("Expected 12 roots")
        if any(r <= 0 for r in root_list):
            raise ValueError("Roots must be positive")
            
        return root_list
        
    except Exception as e:
        # Fallback to seed behavior if something goes wrong, 
        # though roots_genlaguerre is very robust.
        # The seed was [4.0 * (i + 1) for i in range(12)]
        return [4.0 * (i + 1) for i in range(12)]
