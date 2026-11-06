import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    # Grid points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    best_set = []
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    
    # Try multiple restarts with different initial points
    # We will try 10 restarts
    for restart in range(10):
        # Start with an empty set
        selected = []
        # dist_sets: point_tuple -> set of squared distances to other selected points
        dist_sets = {}
        
        # Available points
        available = points.copy()
        
        if len(available) == 0:
            continue
        
        # Pick a random first point
        # To diversify restarts, we can use the restart index to pick different points
        # Or just random.
        idx_start = np.random.randint(len(available))
        p0 = available[idx_start]
        selected.append(p0)
        dist_sets[tuple(p0)] = set()
        
        # Remove p0 from available
        available = np.delete(available, idx_start, axis=0)
        
        while len(available) > 0:
            # Compute distances from all available to all selected
            S_arr = np.array(selected, dtype=np.int32)
            
            # Vectorized distance calculation
            # diff shape: (M, K, 2)
            diff = available[:, np.newaxis, :] - S_arr[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Find min distance for each available point
            min_dists = np.min(dist_sq, axis=1)
            
            # Find the point with max min-distance
            idx_max = np.argmax(min_dists)
            p_star = available[idx_max]
            
            # Check validity of p_star
            valid = True
            d_star = dist_sq[idx_max] # Distances from p_star to all selected
            
            # Check 1: Duplicate distances from p_star
            # If any distance repeats, invalid
            if len(set(d_star)) < len(selected):
                valid = False
            
            # Check 2 & 3: d(p_star, a) == d(a, b) for some b
            if valid:
                p_star_tuple = tuple(p_star)
                for i, a in enumerate(selected):
                    d_pa = d_star[i]
                    a_tuple = tuple(a)
                    # Check if d_pa is in dist_sets[a_tuple]
                    if d_pa in dist_sets[a_tuple]:
                        valid = False
                        break
            
            if valid:
                # Add p_star
                selected.append(p_star)
                p_star_tuple = tuple(p_star)
                dist_sets[p_star_tuple] = set()
                for i, a in enumerate(selected[:-1]):
                    d = d_star[i]
                    a_tuple = tuple(a)
                    dist_sets[p_star_tuple].add(d)
                    dist_sets[a_tuple].add(d)
                
                # Remove from available
                available = np.delete(available, idx_max, axis=0)
            else:
                # Remove from available anyway, as it's invalid for any superset
                available = np.delete(available, idx_max, axis=0)
        
        if len(selected) > len(best_set):
            best_set = selected
    
    # Convert back to list of tuples
    return [tuple(p) for p in best_set]
# EVOLVE-BLOCK-END
