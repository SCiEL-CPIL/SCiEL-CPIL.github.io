import numpy as np
import random
import time

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1)

MAX_DIST_SQ = int(np.max(dist_sq))

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_size = NUM_POINTS + 1
    best_set = None
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Random order
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        S_indices = []
        
        # Re-initialize used_dist for this attempt
        # used_dist[i, d] is True if point i has a neighbor in S at squared distance d
        used_dist_local = np.zeros((NUM_POINTS, MAX_DIST_SQ + 1), dtype=bool)
        
        for p in order:
            if not S_indices:
                # First point is always valid
                S_indices.append(p)
                continue
            
            # Get distances from p to all points in S
            dists = dist_sq[p, S_indices]
            
            # Check 1: Uniqueness of distances from p to S
            if len(dists) != len(np.unique(dists)):
                continue
            
            # Check 2: For each b in S, d(p, b) must not be in used_dist_local[b]
            conflicts = used_dist_local[S_indices, dists]
            
            if np.any(conflicts):
                continue
            
            # Valid! Add p to S
            S_indices.append(p)
            
            # Update used_dist_local
            # For p: mark distances to S as used
            used_dist_local[p, dists] = True
            
            # For each b in S (old S), mark distance to p as used
            used_dist_local[S_indices[:-1], dists] = True
        
        current_size = len(S_indices)
        
        if current_size < best_size:
            best_size = current_size
            best_set = S_indices.copy()
            
            # Early exit if we find a very small set
            if best_size <= 12:
                break

    if best_set is None:
        return []
    
    return [GRID_POINTS[i] for i in best_set]

if __name__ == "__main__":
    # Test run
    res = construct()
    print(f"Found set of size {len(res)}")
