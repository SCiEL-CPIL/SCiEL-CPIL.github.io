import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 110
    # Moore bound for k=3, g=16 is 512
    # LUW has ~118000 vertices
    # We try to find a smaller one
    
    # Fallback: LUW
    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # ... (copy LUW code)
        # Return n, edges

    # Search
    for n in range(512, 2000, 32): # Try sizes
        if time.time() > deadline - 10:
            break
        if try_random_search(n, k, g, deadline - 10):
            return n, edges_found # Need to return edges

    return luw_construct()
