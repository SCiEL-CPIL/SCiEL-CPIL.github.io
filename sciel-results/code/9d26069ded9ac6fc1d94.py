import random
import time

N = 64
TOTAL_POINTS = N * N

def construct():
    coords = [(x, y) for x in range(N) for y in range(N)]
    
    # Precompute squared distances
    # Use a list of arrays for better cache performance? Or just list of lists.
    dist2 = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    for i in range(TOTAL_POINTS):
        x1, y1 = coords[i]
        for j in range(TOTAL_POINTS):
            x2, y2 = coords[j]
            dist2[i][j] = (x1 - x2)**2 + (y1 - y2)**2

    random.seed(42)
    
    def greedy_construct():
        current_set = []
        dist_sets = []
        indices = list(range(TOTAL_POINTS))
        random.shuffle(indices)
        
        for idx in indices:
            # Check can_add
            dists_from_q = []
            valid = True
            
            # 1. Check distances from q to existing points are distinct
            for p in current_set:
                d = dist2[idx][p]
                dists_from_q.append(d)
                
            if len(dists_from_q) != len(set(dists_from_q)):
                continue
                
            # 2. Check if d(p, q) is already in dist_sets[i] for any p
            for i, p in enumerate(current_set):
                d_pq = dist2[p][idx]
                if d_pq in dist_sets[i]:
                    valid = False
                    break
                    
            if valid:
                current_set.append(idx)
                dist_sets.append(set(dists_from_q))
                
                # Update existing dist_sets
                for i, p in enumerate(current_set[:-1]):
                    dist_sets[i].add(dist2[p][idx])
                    
        return current_set, dist_sets

    best_set = []
    best_dist_sets = []
    
    # Initial greedy runs
    for _ in range(20):
        c_set, c_dist = greedy_construct()
        if len(c_set) > len(best_set):
            best_set = c_set
            best_dist_sets = c_dist
            
    current_set = best_set[:]
    current_dist_sets = best_dist_sets[:]
    
    start_time = time.time()
    time_limit = 290
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to add a point
        # We can iterate through all points not in set
        # To speed up, we can keep a list of candidates?
        # For now, just loop all.
        
        in_set = set(current_set)
        
        added = False
        for idx in range(TOTAL_POINTS):
            if idx in in_set:
                continue
                
            dists_from_q = []
            valid = True
            
            # 1. Distances from q to existing points distinct
            for p in current_set:
                d = dist2[idx][p]
                dists_from_q.append(d)
                
            if len(dists_from_q) != len(set(dists_from_q)):
                continue
                
            # 2. Check conflicts
            for i, p in enumerate(current_set):
                d_pq = dist2[p][idx]
                if d_pq in current_dist_sets[i]:
                    valid = False
                    break
                    
            if valid:
                # Add idx
                current_set.append(idx)
                in_set.add(idx)
                current_dist_sets.append(set(dists_from_q))
                
                for i, p in enumerate(current_set[:-1]):
                    current_dist_sets[i].add(dist2[p][idx])
                    
                improved = True
                added = True
                break # Restart add loop to try adding more
                
        if not added:
            # Try swap
            if current_set:
                # Pick random p to remove
                p_idx = random.choice(current_set)
                p_pos = current_set.index(p_idx)
                
                # Remove p_idx
                # We need to update dist_sets
                # This is tricky with lists. 
                # Simpler: Just rebuild dist_sets if we do a swap?
                # Or implement proper remove.
                
                # Let's just do a simple swap by rebuilding if needed, 
                # or just try to find a q that fits if p is gone.
                
                # To keep code simple and fast, let's try a few random swaps
                # by temporarily removing and checking.
                
                found_swap = False
                for _ in range(20):
                    p_remove = random.choice(current_set)
                    
                    # Temporarily remove
                    pos = current_set.index(p_remove)
                    removed_set = current_set[:]
                    removed_set.pop(pos)
                    
                    # We need to check if any q can be added to removed_set
                    # We can reuse the add logic but on removed_set
                    # To avoid full rebuild, we can just check candidates against removed_set
                    # But we need dist_sets for removed_set.
                    # Rebuilding dist_sets for size 10 is fast.
                    
                    # Build dist_sets for removed_set
                    temp_dist_sets = []
                    for i, p in enumerate(removed_set):
                        ds = set()
                        for j, r in enumerate(removed_set):
                            if i != j:
                                ds.add(dist2[p][r])
                        temp_dist_sets.append(ds)
                        
                    # Try to add a random q
                    for _ in range(50):
                        q = random.randint(0, TOTAL_POINTS - 1)
                        if q in removed_set:
                            continue
                            
                        # Check can_add on removed_set
                        dists_from_q = []
                        valid = True
                        for p in removed_set:
                            d = dist2[q][p]
                            dists_from_q.append(d)
                            
                        if len(dists_from_q) != len(set(dists_from_q)):
                            continue
                            
                        for i, p in enumerate(removed_set):
                            d_pq = dist2[p][q]
                            if d_pq in temp_dist_sets[i]:
                                valid = False
                                break
                                
                        if valid:
                            # Accept swap
                            current_set = removed_set
                            current_set.append(q)
                            
                            # Rebuild dist_sets for new current_set
                            current_dist_sets = []
                            for i, p in enumerate(current_set):
                                ds = set()
                                for j, r in enumerate(current_set):
                                    if i != j:
                                        ds.add(dist2[p][r])
                                current_dist_sets.append(ds)
                                
                            found_swap = True
                            improved = True
                            break
                            
                    if found_swap:
                        break
                        
                if not found_swap:
                    # Stuck, do a random restart of local search?
                    # Or just continue
                    pass
                    
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            best_dist_sets = current_dist_sets[:]

    return [coords[i] for i in best_set]
