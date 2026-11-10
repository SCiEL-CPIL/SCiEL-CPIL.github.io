import numpy as np
import random
import time

N = 64
TOTAL_POINTS = N * N

def construct():
    # Precompute coordinates
    coords = np.array([(x, y) for x in range(N) for y in range(N)])
    
    # Compute dist2 using broadcasting for speed
    # coords shape (4096, 2)
    # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # dist2 = np.sum(diff**2, axis=2)
    # This creates a 4096x4096 array.
    # To save memory/time, we can use int16 or int32.
    # Max dist^2 = (63^2 + 63^2) * 2? No, max dx=63, dy=63. 63^2+63^2 = 7938.
    # int16 max is 32767. So int16 is safe and saves memory.
    
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Compute dist2
    # Using broadcasting creates a large temporary array.
    # Let's do it in chunks or just accept the memory hit if it's fast enough.
    # 4096 * 4096 * 2 bytes = ~32MB. This is fine.
    
    # dist2[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    # We can compute this as:
    # x_sq = x_coords**2
    # y_sq = y_coords**2
    # dist2 = x_sq[:, None] + x_sq[None, :] - 2 * np.outer(x_coords, x_coords) + y_sq[:, None] + y_sq[None, :] - 2 * np.outer(y_coords, y_coords)
    # This is more efficient than broadcasting subtraction.
    
    x_sq = x_coords**2
    y_sq = y_coords**2
    dist2 = (x_sq[:, None] + x_sq[None, :] - 2 * np.outer(x_coords, x_coords) + 
             y_sq[:, None] + y_sq[None, :] - 2 * np.outer(y_coords, y_coords)).astype(np.int32)
             
    random.seed(42)
    
    def local_search(start_set, time_limit):
        current_set = list(start_set)
        in_set = set(current_set)
        
        def get_dist_sets(s):
            ds = []
            for i, p in enumerate(s):
                d_set = set()
                for j, q in enumerate(s):
                    if i != j:
                        d_set.add(dist2[p][q])
                ds.append(d_set)
            return ds

        current_dist_sets = get_dist_sets(current_set)
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add a point (Greedy Add)
            added = False
            # Iterate through points not in set
            for idx in range(TOTAL_POINTS):
                if idx in in_set:
                    continue
                
                d_vals = dist2[idx, current_set]
                
                # Check distinctness among new point's distances
                if len(d_vals) != len(set(d_vals)):
                    continue
                
                # Check conflicts with existing dist sets
                valid = True
                for i, p in enumerate(current_set):
                    if d_vals[i] in current_dist_sets[i]:
                        valid = False
                        break
                
                if valid:
                    # Add idx
                    current_set.append(idx)
                    in_set.add(idx)
                    current_dist_sets.append(set(d_vals))
                    
                    # Update existing dist sets
                    for i, p in enumerate(current_set[:-1]):
                        current_dist_sets[i].add(dist2[p][idx])
                        
                    improved = True
                    added = True
                    break # Restart add loop
            
            if not added:
                # 2. Try Remove-Add (Swap)
                # For each point in the set, try removing it and adding any other point
                found_swap = False
                for i, p_remove in enumerate(current_set):
                    # Temporarily remove
                    removed_set = current_set[:i] + current_set[i+1:]
                    
                    if not removed_set:
                        continue
                        
                    temp_dist_sets = get_dist_sets(removed_set)
                    
                    # Try to add any point not in removed_set
                    for q in range(TOTAL_POINTS):
                        if q in removed_set:
                            continue
                            
                        d_vals = dist2[q, removed_set]
                        if len(d_vals) != len(set(d_vals)):
                            continue
                            
                        valid = True
                        for k, p in enumerate(removed_set):
                            if d_vals[k] in temp_dist_sets[k]:
                                valid = False
                                break
                            
                        if valid:
                            # Accept swap
                            current_set = removed_set
                            current_set.append(q)
                            in_set = set(current_set)
                            current_dist_sets = get_dist_sets(current_set)
                            found_swap = True
                            improved = True
                            break
                    
                    if found_swap:
                        break

                    # If no swap found for this removal, we could try to just remove it (perturbation)
                    # But usually we want to maintain size or increase it.
                    # If we can't swap, we might try removing a random point to escape local opt.
                    # However, "Remove-Add" that fails to add is just a removal.
                    # Let's allow removal if we are stuck, but only if we've tried swaps.
                    
                if not found_swap and len(current_set) > 2:
                    # Perturbation: Remove a random point
                    p_remove = random.choice(current_set)
                    pos = current_set.index(p_remove)
                    current_set.pop(pos)
                    in_set = set(current_set)
                    current_dist_sets = get_dist_sets(current_set)
                    improved = True # Force loop to continue

        return current_set

    # Initial Greedy
    best_set = []
    for _ in range(10): # Increase restarts
        indices = list(range(TOTAL_POINTS))
        random.shuffle(indices)
        current_set = []
        in_set = set()
        current_dist_sets = []
        
        for idx in indices:
            if not current_set:
                current_set.append(idx)
                in_set.add(idx)
                current_dist_sets.append(set())
                continue
                
            d_vals = dist2[idx, current_set]
            if len(d_vals) != len(set(d_vals)):
                continue
            
            valid = True
            for i, p in enumerate(current_set):
                if d_vals[i] in current_dist_sets[i]:
                    valid = False
                    break
            
            if valid:
                current_set.append(idx)
                in_set.add(idx)
                current_dist_sets.append(set(d_vals))
                for i, p in enumerate(current_set[:-1]):
                    current_dist_sets[i].add(dist2[p][idx])
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]

    # Local Search from best_set
    final_set = local_search(best_set, 290)
    
    return [tuple(coords[i]) for i in final_set]
