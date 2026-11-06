import random
import time
import numpy as np

N = 64

def construct():
    # Generate all grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Precompute squared distances between all pairs
    # Using numpy for speed
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    # dist_matrix[i][j] is squared distance between points[i] and points[j]
    # Shape (num_points, num_points)
    # dx = xs[:, None] - xs[None, :]
    # dy = ys[:, None] - ys[None, :]
    # dist_matrix = dx*dx + dy*dy
    # This creates a 4096x4096 matrix. ~16M elements.
    # int32 is sufficient (max dist^2 = 2*63^2 = 7938)
    
    # To save memory and time in Python loops, we might stick to lists if NumPy overhead in indexing is high.
    # But NumPy indexing is fast. Let's use a list of lists for easier access in the inner loop if needed, 
    # or just use the NumPy array.
    
    # Actually, pure Python lists of ints might be faster for the tight inner loop 
    # if we avoid NumPy overhead for single element access? 
    # NumPy scalar access is slow. 
    # Let's precompute as a list of lists of ints.
    
    dist_matrix = [[0]*num_points for _ in range(num_points)]
    for i in range(num_points):
        xi, yi = points[i]
        for j in range(i+1, num_points):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx*dx + dy*dy
            dist_matrix[i][j] = d2
            dist_matrix[j][i] = d2
            
    best_set = []
    deadline = time.time() + 290.0
    
    # Indices for all points
    all_indices = list(range(num_points))
    
    while time.time() < deadline:
        current_set = []
        # dist_sets[s] contains the set of squared distances from s to all other points in current_set
        dist_sets = {}
        
        candidates = all_indices[:]
        random.shuffle(candidates)
        
        for idx in candidates:
            # Check if adding idx creates an isosceles triangle
            # 1. Check if idx forms isosceles with any two points in current_set (vertex at idx)
            #    i.e., are there s1, s2 in current_set such that dist(idx, s1) == dist(idx, s2)?
            d_to_s = []
            for s in current_set:
                d_to_s.append(dist_matrix[idx][s])
            
            if len(set(d_to_s)) != len(d_to_s):
                continue # Conflict: isosceles with vertex at idx
            
            # 2. Check if idx forms isosceles with one point in current_set and one other point in current_set
            #    i.e., is there s in current_set such that dist(idx, s) == dist(s, t) for some t in current_set?
            #    Or dist(idx, t) == dist(s, t)?
            #    This is equivalent to: for any s in current_set, is dist(idx, s) in dist_sets[s]?
            conflict = False
            for s in current_set:
                d_idx_s = dist_matrix[idx][s]
                if d_idx_s in dist_sets[s]:
                    conflict = True
                    break
            
            if not conflict:
                # Add idx to current_set
                current_set.append(idx)
                # Update dist_sets
                # For the new point idx, its distances to existing points are d_to_s
                dist_sets[idx] = set(d_to_s)
                # For each existing point s, add the distance to idx
                for s in current_set[:-1]:
                    dist_sets[s].add(dist_matrix[idx][s])
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            # If we found a very large set, we might want to break early or increase time limit?
            # No, keep searching for better.
            
    # Convert indices back to points
    return [points[i] for i in best_set]
