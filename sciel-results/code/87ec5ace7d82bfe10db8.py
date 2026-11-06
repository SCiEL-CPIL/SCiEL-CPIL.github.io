"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    # Generate all grid points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    num_points = len(points)
    
    # Precompute all points as a list of tuples for return, but use numpy for computation
    points_list = [tuple(p) for p in points]
    
    best_set = []
    
    # Number of restarts. We can afford more with vectorized checks.
    num_restarts = 100
    
    # Seed for reproducibility
    random.seed(42)
    
    for _ in range(num_restarts):
        # Shuffle indices
        indices = np.arange(num_points)
        np.random.shuffle(indices)
        
        curr_selected_indices = []
        curr_selected_points = [] # List of numpy arrays
        # dist_sets: dict mapping index to set of squared distances to other selected points
        dist_sets = {}
        
        for idx in indices:
            p = points[idx]
            
            # Check validity against curr_selected_points
            if not curr_selected_points:
                valid = True
            else:
                valid = True
                # Vectorized distance calculation
                # curr_selected_points is a list, convert to array for subtraction?
                # To avoid creating a new array every time, we can keep it as a list and loop,
                # or maintain a numpy array and append.
                # Since K is small, let's just use a list and loop, but optimize the inner check.
                
                # Actually, let's maintain curr_selected_points as a numpy array for vectorization
                # We need to be careful with appending.
                # Let's stick to list of points for simplicity in this step, 
                # but use numpy for the distance vector if we convert.
                
                # Convert current selected to array for vectorized distance
                sel_arr = np.array(curr_selected_points)
                diffs = sel_arr - p
                dists = np.sum(diffs**2, axis=1)
                
                # Check 1: Duplicate distances from p
                # If any distance appears more than once, invalid
                if len(np.unique(dists)) < len(dists):
                    valid = False
                else:
                    # Check 2: Isosceles with base in selected
                    # For each selected point j, check if dist(p, j) is in dist_sets[j]
                    for j in range(len(curr_selected_points)):
                        d_pj = dists[j]
                        sel_idx_j = curr_selected_indices[j]
                        if d_pj in dist_sets[sel_idx_j]:
                            valid = False
                            break
            
            if valid:
                # Add p to selected
                curr_selected_indices.append(idx)
                curr_selected_points.append(p)
                dist_sets[idx] = set()
                
                # Update dist_sets for all existing selected points
                # Calculate distances from p to all existing
                if len(curr_selected_points) > 1:
                    # p is the last one
                    sel_arr = np.array(curr_selected_points[:-1])
                    diffs = sel_arr - p
                    dists = np.sum(diffs**2, axis=1)
                    
                    for j in range(len(curr_selected_points) - 1):
                        d = dists[j]
                        sel_idx_j = curr_selected_indices[j]
                        dist_sets[idx].add(d)
                        dist_sets[sel_idx_j].add(d)
        
        if len(curr_selected_points) > len(best_set):
            best_set = [points_list[idx] for idx in curr_selected_indices]
            
    return best_set

# EVOLVE-BLOCK-END
