<<<<<<< SEARCH
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    if k == 4 and g == 10:
        # Try to construct a smaller graph using random regular graph + 2-switches
        # Moore bound for k=4, g=10 is 243.
        # Known best is 320.
        # We try N from 243 to 400.
        for n_attempt in range(243, 401):
            if time.time() > deadline - 10:
                break
            try:
                n, edges = _random_regular_high_girth(k, g, n_attempt, deadline)
                if n is not None:
                    return n, edges
            except Exception:
                pass
        
        # If random construction failed or took too long, fall back to LUW
        return _luw_graph(k, g)
    else:
        # For other parameters, use LUW as a safe fallback
        return _luw_graph(k, g)
=======
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    if k == 4 and g == 10:
        # Known best is 320. Try to find something smaller.
        # Start with sizes that are multiples of 4 (needed for 4-regular simple graph)
        # and close to the Moore bound.
        candidate_sizes = [320, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400]
        
        # First try the known best size
        for n_attempt in candidate_sizes:
            if time.time() > deadline - 15:
                break
            try:
                n, edges = _random_regular_high_girth(k, g, n_attempt, deadline)
                if n is not None:
                    return n, edges
            except Exception:
                pass
        
        # If random construction failed, fall back to LUW
        return _luw_graph(k, g)
    else:
        # For other parameters, use LUW as a safe fallback
        return _luw_graph(k, g)
>>>>>>> REPLACE

