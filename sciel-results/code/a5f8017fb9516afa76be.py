import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s limit
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Precompute squared distances between all pairs
    # dist[i][j] = squared distance between point i and point j
    n_pts = len(all_points)
    # Use a 2D array for squared distances
    dist = np.zeros((n_pts, n_pts), dtype=np.int32)
    for i in range(n_pts):
        xi, yi = all_points[i]
        for j in range(i+1, n_pts):
            xj, yj = all_points[j]
            d = (xi-xj)**2 + (yi-yj)**2
            dist[i,j] = d
            dist[j,i] = d
    
    best_set = None
    best_size = 0
    
    # Greedy randomized approach with multiple restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random permutation of points
        order = list(range(n_pts))
        random.shuffle(order)
        
        selected = []
        selected_set = set()
        # For each selected point, maintain set of squared distances to other selected points
        dist_sets = {}  # index -> set of squared distances to other selected points
        
        for idx in order:
            if time.time() > deadline:
                break
            
            # Check if adding point idx creates isosceles triangle
            conflict = False
            
            # Check 1: Is idx equidistant to any two selected points?
            # Compute distances from idx to all selected points
            dists_to_selected = []
            for sel_idx in selected:
                d = dist[idx, sel_idx]
                dists_to_selected.append(d)
            
            # Check for duplicate distances (idx equidistant to two points)
            if len(dists_to_selected) != len(set(dists_to_selected)):
                conflict = True
            
            # Check 2: For each selected point, is d(idx, sel) already in sel's distance set?
            if not conflict:
                for sel_idx in selected:
                    d = dist[idx, sel_idx]
                    if d in dist_sets[sel_idx]:
                        conflict = True
                        break
            
            if not conflict:
                # Add point idx
                selected.append(idx)
                selected_set.add(idx)
                dist_sets[idx] = set()
                # Update distance sets for all existing selected points
                for sel_idx in selected[:-1]:
                    d = dist[idx, sel_idx]
                    dist_sets[idx].add(d)
                    dist_sets[sel_idx].add(d)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
        
        # Early termination if we can't improve
        if best_size >= 300:
            break
    
    # Convert back to (x,y) tuples
    if best_set is None:
        return []
    
    return [all_points[i] for i in best_set]
