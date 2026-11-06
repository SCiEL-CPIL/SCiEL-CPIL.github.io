"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    # Generate grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Precompute coordinates
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    # conflict_mask[i] is a bitset where bit j is 1 if i and j conflict
    conflict_mask = [0] * num_points
    
    # Build conflict graph using NumPy for distance grouping
    for k in range(num_points):
        xk = xs[k]
        yk = ys[k]
        
        # Vectorized distance calculation
        dxs = xs - xk
        dys = ys - yk
        d2s = dxs**2 + dys**2
        
        # Sort indices by distance
        sorted_indices = np.argsort(d2s)
        sorted_d2s = d2s[sorted_indices]
        
        # Iterate through groups of equal distance
        i = 0
        while i < num_points:
            d_val = sorted_d2s[i]
            if d_val == 0:
                i += 1
                continue
            
            j = i
            while j < num_points and sorted_d2s[j] == d_val:
                j += 1
            
            # Group is sorted_indices[i:j]
            if j - i >= 2:
                group_indices = sorted_indices[i:j]
                
                # Build bitmask for this group
                group_mask = 0
                for idx in group_indices:
                    group_mask |= (1 << int(idx))
                
                # Update conflict masks
                for u in group_indices:
                    u_int = int(u)
                    conflict_mask[u_int] |= (group_mask ^ (1 << u_int))
            
            i = j

    # Solve MIS using Randomized Greedy with Restarts
    best_solution = []
    num_restarts = 2000
    
    all_indices = list(range(num_points))
    
    for r in range(num_restarts):
        random.shuffle(all_indices)
        selected = []
        selected_bits = 0
        
        for p in all_indices:
            # Check if p conflicts with any selected node
            if (conflict_mask[p] & selected_bits) == 0:
                selected.append(p)
                selected_bits |= (1 << p)
        
        if len(selected) > len(best_solution):
            best_solution = selected
            
    return [points[i] for i in best_solution]

# EVOLVE-BLOCK-END
