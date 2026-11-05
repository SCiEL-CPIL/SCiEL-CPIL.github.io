import time
import random
import math
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)

dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = []
    best_size = 0
    
    all_indices = list(range(TOTAL))
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        random.shuffle(all_indices)
        
        si = []
        ud = []  # list of sets: ud[i] = set of squared distances from si[i] to other points in si
        
        for p_idx in all_indices:
            if not si:
                si.append(p_idx)
                ud.append(set())
                continue
            
            k = len(si)
            
            # Get distances from p to all selected points
            # For small k, manual loop is faster; for larger k, numpy fancy indexing
            if k < 8:
                dists = []
                for q_idx in si:
                    dists.append(dist2[p_idx, q_idx])
            else:
                dists = dist2[p_idx, si].tolist()
            
            # Check 1: All distances from p to si must be distinct
            if len(set(dists)) != k:
                continue
            
            # Check 2: For each existing point, the distance to p is not already used
            valid = True
            for i in range(k):
                if dists[i] in ud[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add p to the set
            si.append(p_idx)
            ud.append(set(dists))
            
            # Update existing points' distance sets
            for i in range(k):
                ud[i].add(dists[i])
        
        # Refinement: Try to add more points that were rejected
        # Build a set of selected indices for O(1) lookup
        selected_set = set(si)
        
        # Try each point not in si to see if it can be added
        # We do this in a random order to potentially find improvements
        unselected = [p for p in all_indices if p not in selected_set]
        random.shuffle(unselected)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for p_idx in unselected:
                if p_idx in selected_set:
                    continue
                
                k = len(si)
                dists = dist2[p_idx, si].tolist()
                
                if len(set(dists)) != k:
                    continue
                
                valid = True
                for i in range(k):
                    if dists[i] in ud[i]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Can add this point
                si.append(p_idx)
                selected_set.add(p_idx)
                ud.append(set(dists))
                for i in range(k):
                    ud[i].add(dists[i])
                unselected.remove(p_idx)
                improved = True
                break  # Restart the loop since ud changed
        
        # Update best
        if len(si) > best_size:
            best_size = len(si)
            best_set = [(idx // N, idx % N) for idx in si]
        
        # Early termination if we've used most of our time
        if time.time() > deadline - 30:
            break
    
    return best_set
