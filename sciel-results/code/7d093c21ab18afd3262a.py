import numpy as np

def priority(vector, n):
    # Convert vector to numpy array for efficient counting
    # vector is a tuple of ints
    v = np.array(vector)
    
    # Heuristic: Prioritize vectors with a high count of 2s.
    # This is a static approximation of a "conflict-minimizing" order.
    # Vectors with many 2s are structurally distinct from those with many 1s.
    # By grouping 2s, we might create a more homogeneous set that avoids lines
    # more effectively than a mixed or random order.
    
    count_2s = np.sum(v == 2)
    
    # Secondary tie-breaker: Prefer higher total weight (more non-zeros)
    # to push away from the origin which is a "hub" of conflicts?
    # Or prefer lower weight?
    # Let's add a small epsilon based on weight to break ties.
    weight = np.sum(v != 0)
    
    return float(count_2s * 100 + weight)
