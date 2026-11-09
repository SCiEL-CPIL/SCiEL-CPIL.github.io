# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    maximizing the number of points such that the minimum pairwise squared
    distance is greater than the maximum squared norm.
    """
    d = 11
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Start with a set of candidate points with small norms
    # We'll use points with coordinates in {-1, 0, 1} and small integers
    # to keep norms small while allowing good separation
    
    # Generate candidate points: all points with coords in {-2, -1, 0, 1, 2}
    # but filter to those with reasonable norm
    coords = [-2, -1, 0, 1, 2]
    
    # Build candidate list with their norms
    candidates = []
    for point in product(coords, repeat=d):
        p = np.array(point, dtype=np.int64)
        norm_sq = int(np.sum(p * p))
        if norm_sq == 0:
            continue
        candidates.append((p, norm_sq))
    
    # Sort by norm (smaller norms are easier to add)
    candidates.sort(key=lambda x: x[1])
    
    # Greedy construction: maintain a list of selected points
    # Constraint: for all pairs (i,j), ||pi - pj||^2 > max_norm_sq
    # where max_norm_sq = max over all selected points of ||p||^2
    
    selected = []
    selected_norms = []
    
    for p, norm_sq in candidates:
        if time.time() > deadline:
            break
        
        # Check if adding p violates constraints
        # New max norm would be max(current_max, norm_sq)
        new_max_norm = max(selected_norms) if selected_norms else 0
        if norm_sq > new_max_norm:
            new_max_norm = norm_sq
        
        # Check all existing points: distance squared must be > new_max_norm
        valid = True
        for sp in selected:
            diff = p - sp
            dist_sq = int(np.sum(diff * diff))
            if dist_sq <= new_max_norm:
                valid = False
                break
        
        if valid:
            selected.append(p)
            selected_norms.append(norm_sq)
    
    # Convert to array
    if not selected:
        # Fallback: at least two points
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    else:
        points = np.array(selected, dtype=np.int64)
    
    # Ensure shape is (k, 11) and round/cast
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
