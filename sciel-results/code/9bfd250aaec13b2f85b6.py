import time
import random
import math
import numpy as np

N = 64
TIME_LIMIT = 290 # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Precompute all points
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    num_points = len(all_points)
    
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def dist2_vec(p, pts):
        # p: (2,), pts: (K, 2)
        # returns (K,) array of squared distances
        diff = pts - p
        return np.sum(diff * diff, axis=1)
        
    def rebuild_dist_sets(selected):
        # selected: list of point indices
        # returns list of sets of distances
        K = len(selected)
        dist_sets = [set() for _ in range(K)]
        for i in range(K):
            pi = all_points[selected[i]]
            for j in range(K):
                if i == j: continue
                pj = all_points[selected[j]]
                d = dist2_vec(pi, pj.reshape(1,2))[0]
                dist_sets[i].add(int(d))
        return dist_sets

    def try_add(p_idx, selected, dist_sets):
        # selected: list of indices
        # dist_sets: list of sets
        # Returns True if added, and updates selected, dist_sets
        p = all_points[p_idx]
        K = len(selected)
        
        if K == 0:
            selected.append(p_idx)
            dist_sets.append(set())
            return True
            
        # Calculate distances from p to all in selected
        # selected_indices = np.array(selected)
        # sel_pts = all_points[selected_indices]
        # d_p = dist2_vec(p, sel_pts)
        
        # For speed, we do it in a loop if K is small, or vectorized if K is large.
        # K is expected to be small (< 20).
        d_p = []
        for idx in selected:
            pt = all_points[idx]
            d = dist2_vec(p, pt.reshape(1,2))[0]
            d_p.append(int(d))
            
        # Check 1: Distances from p to others must be distinct
        if len(set(d_p)) != len(d_p):
            return False
            
        # Check 2: For each existing point i, d_p[i] must not be in dist_sets[i]
        for i in range(K):
            if d_p[i] in dist_sets[i]:
                return False
                
        # Add p
        selected.append(p_idx)
        dist_sets.append(set(d_p))
        
        # Update existing dist_sets
        for i in range(K):
            dist_sets[i].add(d_p[i])
            
        return True

    def try_remove(idx, selected, dist_sets):
        # Remove point at index idx
        p_idx = selected[idx]
        p = all_points[p_idx]
        K = len(selected)
        
        # Remove distances from other sets
        for i in range(K):
            if i == idx: continue
            pt = all_points[selected[i]]
            d = dist2_vec(p, pt.reshape(1,2))[0]
            dist_sets[i].discard(int(d))
            
        selected.pop(idx)
        dist_sets.pop(idx)

    def systematic_local_search(initial_selected, initial_dist_sets):
        selected = list(initial_selected)
        dist_sets = [set(s) for s in initial_dist_sets]
        current_size = len(selected)
        selected_set = set(selected)
        
        local_best = list(selected)
        local_best_size = current_size
        
        # Max iterations to avoid infinite loops in case of bugs
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # 1. Try to add any point (Greedy)
            added = False
            # Shuffle candidates to break symmetry
            candidates = [i for i in range(num_points) if i not in selected_set]
            rng.shuffle(candidates)
            for p_idx in candidates:
                if try_add(p_idx, selected, dist_sets):
                    selected_set.add(p_idx)
                    current_size += 1
                    added = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best = list(selected)
                    break # Restart greedy phase
                    
            if added:
                continue
                
            # 2. Try 1-Swaps (Remove u, Add v)
            # Try all u in selected, all v not in selected
            # This is O(K * (N^2 - K))
            # We can optimize by pre-filtering?
            # For now, just iterate.
            
            swapped = False
            # Shuffle selected indices to vary the order
            u_indices = list(range(current_size))
            rng.shuffle(u_indices)
            
            for i in u_indices:
                u_idx = selected[i]
                
                # Try to remove u
                # We need to check if removing u allows adding some v
                # Instead of removing and restoring, we can simulate?
                # Simulation is hard. Let's remove, try adds, restore if fail.
                
                # Backup
                selected_backup = list(selected)
                dist_sets_backup = [set(s) for s in dist_sets]
                selected_set_backup = set(selected_set)
                
                try_remove(i, selected, dist_sets)
                selected_set.discard(u_idx)
                current_size -= 1
                
                # Try to add any v
                candidates = [j for j in range(num_points) if j not in selected_set]
                rng.shuffle(candidates)
                
                success = False
                for v_idx in candidates:
                    if try_add(v_idx, selected, dist_sets):
                        selected_set.add(v_idx)
                        current_size += 1
                        success = True
                        # Accepted swap
                        # Note: size is same, so local_best_size doesn't change
                        # But the state changed, which might allow future adds
                        break
                        
                if not success:
                    # Restore
                    selected = selected_backup
                    dist_sets = dist_sets_backup
                    selected_set = selected_set_backup
                    current_size += 1
                else:
                    swapped = True
                    break # Restart search with new state
            
            if swapped:
                continue
                
            # 3. If stuck, try 2-Swaps? Or break and restart?
            # 2-Swaps are expensive. Let's just break and let the main loop restart.
            # Or do a random 2-swap.
            
            if current_size >= 2:
                 # Random 2-swap
                 i, j = rng.sample(range(current_size), 2)
                 u_idx = selected[i]
                 v_idx = selected[j]
                 
                 selected_backup = list(selected)
                 dist_sets_backup = [set(s) for s in dist_sets]
                 selected_set_backup = set(selected_set)
                 size_backup = current_size
                 
                 # Remove i and j
                 # Remove higher index first
                 if i > j:
                     i, j = j, i
                 try_remove(i, selected, dist_sets)
                 selected_set.discard(selected[i]) # Wait, index changed?
                 # Let's be careful.
                 # Remove j first if j > i?
                 # Let's just use the backup/restore approach carefully.
                 
                 # Re-do removal logic safely
                 # Restore state first
                 selected = selected_backup
                 dist_sets = dist_sets_backup
                 selected_set = selected_set_backup
                 current_size = size_backup
                 
                 # Remove i
                 try_remove(i, selected, dist_sets)
                 selected_set.discard(selected[i]) # This is wrong, selected[i] changed after pop?
                 # try_remove pops. So selected[i] is gone.
                 # We need the value before pop.
                 
                 # Let's rewrite try_remove to return the removed item? Or just store it.
                 # In the 1-swap above, I used u_idx = selected[i] BEFORE try_remove.
                 # Here:
                 u_idx = selected[i]
                 v_idx = selected[j]
                 
                 if i > j:
                     i, j = j, i
                     u_idx, v_idx = v_idx, u_idx
                     
                 # Remove j (lower index)
                 try_remove(j, selected, dist_sets)
                 selected_set.discard(v_idx)
                 current_size -= 1
                 
                 # Adjust i
                 if i > j:
                     i -= 1
                 # Now i is the index of u_idx in the new list
                 
                 # Remove i
                 try_remove(i, selected, dist_sets)
                 selected_set.discard(u_idx)
                 current_size -= 1
                 
                 # Try to add 2 points
                 candidates = [k for k in range(num_points) if k not in selected_set]
                 rng.shuffle(candidates)
                 
                 # Try pairs
                 success = False
                 # Limit pair attempts
                 for a in range(min(50, len(candidates))):
                     w_idx = candidates[a]
                     if try_add(w_idx, selected, dist_sets):
                         selected_set.add(w_idx)
                         current_size += 1
                         for b in range(a+1, min(a+50, len(candidates))):
                             x_idx = candidates[b]
                             if try_add(x_idx, selected, dist_sets):
                                 selected_set.add(x_idx)
                                 current_size += 1
                                 success = True
                                 break
                         if success:
                             break
                         else:
                             # Remove w_idx
                             # Find index of w_idx
                             w_pos = selected.index(w_idx)
                             try_remove(w_pos, selected, dist_sets)
                             selected_set.discard(w_idx)
                             current_size -= 1

                 if not success:
                     # Restore
                     selected = selected_backup
                     dist_sets = dist_sets_backup
                     selected_set = selected_set_backup
                     current_size = size_backup
            else:
                break # Cannot do 2-swap if size < 2

        return local_best, local_best_size

    # Main Loop: Multiple Restarts
    while time.time() < deadline - 10:
        # 1. Generate initial set via Randomized Greedy
        selected = []
        dist_sets = []
        candidates = list(range(num_points))
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if try_add(p_idx, selected, dist_sets):
                pass
            if time.time() > deadline - 5:
                break
        
        current_size = len(selected)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(selected)
        
        # 2. Local Search
        if time.time() < deadline - 20:
            # Rebuild dist_sets to be safe (though try_add maintains it)
            dist_sets = rebuild_dist_sets(selected)
            
            local_best, local_best_size = systematic_local_search(selected, dist_sets)
            
            if local_best_size > best_size:
                best_size = local_best_size
                best_set = list(local_best)

    # Convert indices back to coordinates
    return [tuple(all_points[i]) for i in best_set]

# Test/Verify
if __name__ == "__main__":
    res = construct()
    print(len(res))
