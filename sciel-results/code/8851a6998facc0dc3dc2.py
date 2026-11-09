import time
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_indices = list(range(len(all_points)))
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(p, selected, dist_from):
        px, py = p
        d_p = []
        for i in range(len(selected)):
            ax, ay = selected[i]
            d_sq = (px - ax)**2 + (py - ay)**2
            d_p.append(d_sq)
            
        if len(set(d_p)) != len(d_p):
            return False, None
            
        for i in range(len(selected)):
            if d_p[i] in dist_from[i]:
                return False, None
        return True, d_p

    def add_point(p, selected, dist_from, d_p):
        selected.append(p)
        new_dist_set = set()
        for i in range(len(selected) - 1):
            new_dist_set.add(d_p[i])
        dist_from.append(new_dist_set)
        for i in range(len(selected) - 1):
            dist_from[i].add(d_p[i])

    def remove_point(idx, selected, dist_from):
        p = selected[idx]
        # Remove p from selected
        selected.pop(idx)
        
        # Remove p's dist set
        dist_from.pop(idx)
        
        # Update other dist_from sets
        # We need to remove the distance between p and each other point
        # We don't have the distances stored in a way that allows easy O(1) removal from the set
        # dist_from[i] is a set of distances.
        # We can just recalculate dist_from from scratch? 
        # O(K^2) to rebuild. K is small (~10-20).
        # This is safer and simpler.
        
        # Actually, let's just rebuild dist_from after any modification in the SA loop.
        # It's O(K^2) which is fast.

    def rebuild_dist_from(selected):
        dist_from = []
        for i in range(len(selected)):
            s = set()
            ax, ay = selected[i]
            for j in range(len(selected)):
                if i == j: continue
                bx, by = selected[j]
                s.add((ax-bx)**2 + (ay-by)**2)
            dist_from.append(s)
        return dist_from

    def greedy_select(points_order):
        selected = []
        dist_from = []
        
        for p in points_order:
            valid, d_p = is_valid_addition(p, selected, dist_from)
            if valid:
                add_point(p, selected, dist_from, d_p)
                
        return selected, dist_from

    # Run multiple restarts
    while time.time() < deadline:
        random.shuffle(point_indices)
        order = [all_points[i] for i in point_indices]
        
        current_set, current_dist = greedy_select(order)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Simulated Annealing / Random Walk on maximal sets
        # Try to find a larger set by swapping
        # We only do this if we have time and haven't found a very large set
        # (or just always do it if time permits)
        
        if len(current_set) < best_size:
            continue
            
        # Try to improve current_set
        # Rebuild dist_from to ensure it's consistent
        current_dist = rebuild_dist_from(current_set)
        
        # We maintain a set of selected points for O(1) lookup
        selected_set = set(current_set)
        
        # We want to try to add points. 
        # First, check if any point can be added.
        # Since it's maximal, probably not.
        # So we do swaps.
        
        # To speed up, we can try to add points from a random subset first.
        
        improved = False
        # Number of swaps to try before checking for additions
        swap_attempts = 0
        max_swaps = 1000
        
        while swap_attempts < max_swaps and time.time() < deadline - 5:
            swap_attempts += 1
            
            # Pick a point to remove
            idx_to_remove = random.randint(0, len(current_set) - 1)
            p_remove = current_set[idx_to_remove]
            
            # Pick a point to add
            # Random from all_points not in current_set
            # To avoid O(N^2) scan for 'not in', we can just pick random index and check
            p_add = None
            attempts = 0
            while attempts < 10: # Try a few times to find a valid candidate index
                rand_idx = random.randint(0, len(all_points) - 1)
                candidate = all_points[rand_idx]
                if candidate not in selected_set:
                    p_add = candidate
                    break
                attempts += 1
            
            if p_add is None:
                continue
                
            # Try swap
            # Temporarily remove p_remove
            # We need to check if p_add is valid in current_set - {p_remove}
            
            # Optimization: 
            # Check validity of p_add against current_set - {p_remove}
            # This is O(K).
            
            # Let's create a temporary list
            temp_set = [p for i, p in enumerate(current_set) if i != idx_to_remove]
            
            # Check p_add against temp_set
            valid, d_p_add = is_valid_addition(p_add, temp_set, rebuild_dist_from(temp_set)) # Rebuild is O(K^2), maybe too slow?
            # K is ~10. K^2 is 100. 1000 swaps * 100 ops = 100k ops. Fine.
            
            if valid:
                # Perform swap
                current_set[idx_to_remove] = p_add
                selected_set.remove(p_remove)
                selected_set.add(p_add)
                
                # Rebuild dist_from
                current_dist = rebuild_dist_from(current_set)
                
                # Now, try to add any point
                # Check a random subset of points for addition
                added = False
                for _ in range(50): # Check 50 random points
                    rand_idx = random.randint(0, len(all_points) - 1)
                    candidate = all_points[rand_idx]
                    if candidate not in selected_set:
                        v, d = is_valid_addition(candidate, current_set, current_dist)
                        if v:
                            add_point(candidate, current_set, current_dist, d)
                            selected_set.add(candidate)
                            current_dist = rebuild_dist_from(current_set)
                            added = True
                            break
                
                if added:
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    # Reset swap attempts? Or continue?
                    # If we added a point, we are in a new state.
                    # Continue searching.
                    swap_attempts = 0 
                    max_swaps = 1000
            else:
                # Invalid swap, do nothing
                pass
        
        # End of SA loop for this restart
        
    return best_set
