import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Precompute squared distances
    # dist[i][j] = squared distance between points[i] and points[j]
    # Using a list of lists or a 2D numpy array
    # 4096 x 4096 = 16,777,216 entries.
    # int64 is 8 bytes. 16M * 8 = 134 MB. This might be too much or slow to allocate/init.
    # Let's calculate on the fly or use a smaller structure if possible.
    # Actually, 134MB is fine for modern systems, but let's be safe.
    # We can compute distance in O(1) from coordinates.
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    best_size = 0
    best_set = []
    
    # We will run multiple random restarts
    # Each restart: shuffle points, greedily add valid ones
    
    while time.time() < deadline:
        random.shuffle(points)
        current_set = []
        # dists_from[i] is a set of squared distances from points[i] to other points in current_set
        # We need to map point to its index in current_set? No, just keep a dict or parallel list.
        # Let's keep a list of points and a dict mapping point -> set of dists.
        # Since points are tuples, they are hashable.
        dist_map = {} # point -> set of squared distances to other points in current_set
        
        for p in points:
            # Check if p can be added to current_set
            valid = True
            dists_from_p = []
            
            for q in current_set:
                d2 = sq_dist(p, q)
                dists_from_p.append(d2)
                
                # Condition 1: d2 must not be in dist_map[q]
                if d2 in dist_map[q]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Condition 2: all distances from p to current_set must be distinct
            # If len(current_set) > 0, check for duplicates in dists_from_p
            if len(dists_from_p) != len(set(dists_from_p)):
                continue
                
            # If we get here, p is valid
            # Add p to current_set
            current_set.append(p)
            dist_map[p] = set(dists_from_p)
            
            # Update dist_map for all existing points q
            for i, q in enumerate(current_set[:-1]): # Exclude p itself
                dist_map[q].add(sq_dist(p, q))
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
            # If we found a very large set, we might want to break or continue
            # Theoretical max? 
            # For N=64, if we can get > 100, that's good.
            
    return best_set
