import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 330s limit
    
    # Point representation: index 0..4095
    # x = i // N, y = i % N
    num_points = N * N
    
    # Precompute coordinates for all points to speed up distance calculation
    coords = [(i // N, i % N) for i in range(num_points)]
    
    # Data structures for incremental validity checks
    selected = []
    point_set = set()
    # dist_sets[q] = set of squared distances from q to other selected points
    dist_sets = {}
    
    def _dist2_idx(i, j):
        xi, yi = coords[i]
        xj, yj = coords[j]
        dx = xi - xj
        dy = yi - yj
        return dx*dx + dy*dy

    def _can_add(p_idx):
        if p_idx in point_set:
            return False
        k = len(selected)
        if k == 0:
            return True
        
        seen_dists = set()
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            # Condition 1: Distances from p must be unique (no isosceles with vertex p)
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Condition 2: d must not be in dist_sets[q_idx] (no isosceles with vertex q)
            if d in dist_sets[q_idx]:
                return False
        
        return True

    def _add(p_idx):
        point_set.add(p_idx)
        selected.append(p_idx)
        dist_sets[p_idx] = set()
        
        for q_idx in selected[:-1]: # All except p_idx
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[p_idx].add(d)
            dist_sets[q_idx].add(d)

    def _remove(p_idx):
        point_set.remove(p_idx)
        # Remove from list (O(K), K is small)
        try:
            idx = selected.index(p_idx)
            selected.pop(idx)
        except ValueError:
            pass
        
        # Update dist_sets for all remaining points
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[q_idx].remove(d)
        
        del dist_sets[p_idx]
        
    def _reset_state(best_selected, best_point_set, best_dist_sets):
        selected[:] = best_selected
        point_set.clear()
        point_set.update(best_point_set)
        dist_sets.clear()
        for k, v in best_dist_sets.items():
            dist_sets[k] = set(v)

    def _greedy_init(rng_seed):
        """Run a greedy initialization and return the best state found."""
        nonlocal selected, point_set, dist_sets
        # Clear current state
        selected.clear()
        point_set.clear()
        dist_sets.clear()
        
        rng_init = random.Random(rng_seed)
        candidates = list(range(num_points))
        rng_init.shuffle(candidates)
        
        for p_idx in candidates:
            if _can_add(p_idx):
                _add(p_idx)
                
        return len(selected), selected[:], point_set.copy(), {k: set(v) for k, v in dist_sets.items()}

    # 1. Improved Initialization: Try a few seeds
    best_size = 0
    best_selected = []
    best_point_set = set()
    best_dist_sets = {}
    
    for seed in [42, 123, 999, 555, 777]:
        size, sel, pset, dsets = _greedy_init(seed)
        if size > best_size:
            best_size = size
            best_selected = sel
            best_point_set = pset
            best_dist_sets = dsets
            
    # Load best state into active structures
    _reset_state(best_selected, best_point_set, best_dist_sets)
    
    rng = random.Random(12345)
    
    # 2. Local Search with Swaps
    stuck_count = 0
    max_stuck_before_swap = 20 # Try swaps more frequently
    
    while time.time() < deadline:
        # 1. Try to add points
        added_any = False
        
        # Try a batch of random additions
        for _ in range(100):
            if time.time() > deadline:
                break
                
            p_idx = rng.randint(0, num_points - 1)
            if p_idx in point_set:
                continue
                
            if _can_add(p_idx):
                _add(p_idx)
                added_any = True
                stuck_count = 0
                # If we added, we might be able to add more, so continue
                
        if added_any:
            # Update best if improved
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = selected[:]
                best_point_set = point_set.copy()
                best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
            continue # Go back to start of while loop, try adding more
        
        # 2. If no adds, we are stuck
        stuck_count += 1
        
        # If stuck, try a swap move
        if stuck_count > max_stuck_before_swap:
            if selected:
                # Pick a random point to remove
                remove_idx = rng.randint(0, len(selected) - 1)
                p_idx_to_remove = selected[remove_idx]
                
                # Save state to revert if swap fails
                # We don't need to save the whole state if we just revert the removal
                # But we need to try adding a new point.
                
                _remove(p_idx_to_remove)
                
                swap_success = False
                # Try to find a point to add
                # Instead of random, we can try a few random points
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    q_idx = rng.randint(0, num_points - 1)
                    if q_idx in point_set:
                        continue
                    if _can_add(q_idx):
                        _add(q_idx)
                        swap_success = True
                        break
                
                if not swap_success:
                    # Revert the removal
                    _add(p_idx_to_remove)
                
                # Reset stuck count to encourage more swaps if we are still stuck
                # But if we succeeded, we might have opened up space for an add in the next loop
                stuck_count = 0
                
                # Update best if improved (unlikely in swap, but possible if we added back the same point? No, size is same)
                # However, if we swapped, the structure changed.
                
        # Optional: If stuck for a very long time, restart from best state?
        if stuck_count > 1000:
             # Restore best state
             _reset_state(best_selected, best_point_set, best_dist_sets)
             stuck_count = 0

    # Convert best_selected indices back to (x, y) tuples
    result = []
    for idx in best_selected:
        x = idx // N
        y = idx % N
        result.append((x, y))
        
    return result
# EVOLVE-BLOCK-END
