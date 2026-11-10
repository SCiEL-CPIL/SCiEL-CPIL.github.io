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
        
    def _greedy_repair():
        """Attempt to add any valid points to the current set."""
        # Iterate over all points? No, that's O(N^2 * K).
        # Instead, we can try random points or iterate in a specific order.
        # Since we are stuck, a full sweep might be expensive if K is large, but K is small (~10-20).
        # However, N^2 = 4096. 4096 * 20 = 81920 ops. This is acceptable for a repair step.
        # But doing this every time we are stuck might be too frequent.
        # We'll only do a full sweep if we are stuck for a while.
        
        # For now, let's just do a random sweep in the main loop logic.
        pass

    rng = random.Random(12345)
    
    # 1. Greedy Initialization
    # Try a few different greedy orders to find a good starting point
    best_size = 0
    best_state = None
    
    # We will run the main search loop, but start with a greedy fill
    candidates = list(range(num_points))
    
    # Perform a greedy fill with a fixed seed first to ensure a baseline
    rng_init = random.Random(42)
    rng_init.shuffle(candidates)
    for p_idx in candidates:
        if _can_add(p_idx):
            _add(p_idx)
            
    best_size = len(selected)
    # Save state
    best_selected = selected[:]
    best_point_set = point_set.copy()
    best_dist_sets = {k: set(v) for k, v in dist_sets.items()}

    # 2. Local Search / Simulated Annealing
    # We will try to improve upon the current state.
    
    stuck_count = 0
    max_stuck_before_remove = 50 # Lower threshold to kick earlier
    
    # Helper to find the point with the highest "conflict degree"
    # Degree is the number of unselected points that cannot be added due to this point?
    # That's expensive to compute dynamically.
    # Alternative: Remove the point that is "most central" or has the most edges in the conflict graph?
    # Simpler heuristic: Remove a random point, but if we fail to recover, try removing the one with the most connections in the selected set?
    # Actually, the "conflict" is with unselected points.
    
    # Let's stick to random removal but increase the "kick" effectiveness.
    
    while time.time() < deadline:
        # 1. Try to add points
        added_any = False
        
        # Try a batch of random additions
        # Increase batch size to find more potential adds before declaring stuck
        for _ in range(200):
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
        
        # If stuck for a while, try to remove a point to escape local optimum
        if stuck_count > max_stuck_before_remove:
            if selected:
                # Strategy: Remove a random point.
                # To improve, we can try to remove the point that "blocks" the most other points.
                # Approximation: Count how many unselected points are invalid *specifically* because of this point?
                # That requires checking _can_add for all unselected points, which is O(N^2 * K).
                # Too slow for every kick.
                
                # Alternative: Random removal is okay, but let's try to be smarter about the "kick".
                # Instead of just random re-adds, we do a "Greedy Repair" pass.
                # We will pick a random point to remove.
                remove_idx = rng.randint(0, len(selected) - 1)
                p_idx_to_remove = selected[remove_idx]
                _remove(p_idx_to_remove)
                
                # After removing, we are "unstuck" potentially, so reset counter
                stuck_count = 0
                
                # Immediately try to add points back to see if we can recover
                # This is a "kick"
                # We will do a more thorough repair here:
                # 1. Try to add the removed point back? No, it was removed because it was blocking?
                #    Actually, we removed it to see if we can add *more* or *different* points.
                
                # Let's do a "Greedy Sweep" on a subset of points.
                # We can't sweep all 4096 points efficiently if we do it often.
                # But we can do a few hundred random tries.
                
                kick_success = False
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    p_idx = rng.randint(0, num_points - 1)
                    if p_idx in point_set:
                        continue
                    if _can_add(p_idx):
                        _add(p_idx)
                        kick_success = True
                        # If we added, continue trying to add more
                        # We don't break, we keep trying to fill the space created by removal
                        # If we manage to add more than 1 point, it's a net gain.
                        # If we add exactly 1, it's a neutral swap (size unchanged).
                        # If we add 0, it's a loss (size decreased).
                        
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = selected[:]
                    best_point_set = point_set.copy()
                    best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
                
                # If we ended up with fewer points than before the kick, it might be good to restore?
                # But we don't save the pre-kick state easily without overhead.
                # We rely on the global best restoration later if we get stuck for a long time.
                
        # Optional: If stuck for a very long time, restart from best state?
        # This might be useful if the current state is much worse than best.
        if stuck_count > 500:
             # Restore best state
             selected[:] = best_selected
             point_set.clear()
             point_set.update(best_point_set)
             dist_sets.clear()
             for k, v in best_dist_sets.items():
                 dist_sets[k] = set(v)
             stuck_count = 0

    # Convert best_selected indices back to (x, y) tuples
    result = []
    for idx in best_selected:
        x = idx // N
        y = idx % N
        result.append((x, y))
        
    return result
# EVOLVE-BLOCK-END
