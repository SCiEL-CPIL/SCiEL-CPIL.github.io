import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if k == 4 and g == 10:
        # Get LUW size
        n_luw, edges_luw = _luw_fallback(4, 10)
        
        # Try local search for n < n_luw
        # Moore bound is 242. Start at 320 (known best).
        # Step by 2 (bipartite)
        for n in range(320, n_luw, 2):
            if time.time() > deadline - 10: break # Leave time for return
            
            # Try to find a graph of size n
            success, edges = _local_search(k, g, n, time_limit=5.0)
            if success:
                return n, edges
                
        # Fallback to LUW
        return n_luw, edges_luw

    # Generic fallback
    return _luw_fallback(k, g)

# ... helper functions ...
