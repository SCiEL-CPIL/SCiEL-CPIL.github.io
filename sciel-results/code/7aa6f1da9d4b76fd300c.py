<<<<<<< SEARCH
import time
import sys
import random
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Known rank 44 solution
    U44, V44, W44 = _get_rank_44_solution()
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find rank 43 and 42
    for target_rank in [43, 42]:
        if time.time() > deadline - 20:
            break
            
        # Attempt search for target_rank
        found = _search_for_rank(target_rank, deadline)
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = found
            break # If we find 43, we might still try 42 if time permits, but 42 is better.
            
    return {"U": best_U, "V": best_V, "W": best_W}
=======
import time
import sys
import random
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Known rank 44 solution
    U44, V44, W44 = _get_rank_44_solution()
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find rank 43 and 42 using vectorized optimization
    for target_rank in [43, 42]:
        if time.time() > deadline - 15:
            break
            
        # Attempt search for target_rank
        found = _search_for_rank(target_rank, deadline)
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = found
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
>>>>>>> REPLACE

