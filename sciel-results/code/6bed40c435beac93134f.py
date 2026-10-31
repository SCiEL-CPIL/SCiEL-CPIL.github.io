import math
from functools import lru_cache

# Pre-compute entropy for all possible (c0, c1, c2) combinations
# c0 + c1 + c2 = n
# We don't know n in advance, but we can compute on the fly or cache per n.
# Since n is fixed for a run, we can cache based on (c0, c1, c2, n) or just compute.
# Computing log is cheap.

def priority(vector, n):
        # Count 0s, 1s, 2s
        # vector is a tuple of 0, 1, 2
        c0 = vector.count(0)
        c1 = vector.count(1)
        c2 = n - c0 - c1
        
        # Calculate Shannon Entropy
        # H = - sum (p_i * log2(p_i))
        h = 0.0
        if c0 > 0:
            p0 = c0 / n
            h -= p0 * math.log2(p0)
        if c1 > 0:
            p1 = c1 / n
            h -= p1 * math.log2(p1)
        if c2 > 0:
            p2 = c2 / n
            h -= p2 * math.log2(p2)
            
        return h
