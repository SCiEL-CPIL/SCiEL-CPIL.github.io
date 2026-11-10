import time
import random
import math

N = 64
TIME_LIMIT = 290 # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def rebuild_dist_from(selected):
        dist_from = []
        for i in range(len(selected)):
            s = set()
            pi = selected[i]
            for j in range(len(selected)):
                if i == j: continue
                pj = selected[j]
                s.add(dist2(pi, pj))
            dist_from.append(s)
        return dist_from

    def try_add(p, selected, dist_from):
        d_p = []
        for pt in selected:
            d_p.append(dist2(p, pt))
            
        if len(set(d_p)) != len(d_p):
            return False
            
        for i in range(len(selected)):
            if d_p[i] in dist_from[i]:
                return False
                
        selected.append(p)
        new_dist_set = set(d_p)
        dist_from.append(new_dist_set)
        for i in range(len(selected) - 1):
            dist_from[i].add(d_p[i])
            
        return True

    def try_remove(idx, selected, dist_from):
        p = selected[idx]
        
        # Remove distances involving p from other points' sets
        # We iterate over all other points j
        for j in range(len(selected)):
            if j == idx: continue
            d = dist2(p, selected[j])
            dist_from[j].discard(d)
            
        selected.pop(idx)
        dist_from.pop(idx)

    def local_search(initial_set, initial_dist):
        selected = list(initial_set)
        dist_from = [set(s) for s in initial_dist]
        
        current_size = len(selected)
        local_best = list(selected)
        local_best_size = current_size
        selected_set = set(selected)
        
        # 1. Try to grow
        # 2. If stuck, 1-Swap
        # 3. If stuck, 2-Swap
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            # 1. Try to add a point
            added = False
            # Try a few random candidates to see if we can grow
            for _ in range(50):
                p = rng.choice(all_points)
                if p in selected_set:
                    continue
                if try_add(p, selected, dist_from):
                    selected_set.add(p)
                    current_size += 1
                    added = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best = list(selected)
                    break
            
            if added:
                continue
                
            # 2. Try 1-Swap
            swapped = False
            if current_size > 0:
                idx_u = rng.randint(0, current_size - 1)
                u = selected[idx_u]
                
                try_remove(idx_u, selected, dist_from)
                selected_set.discard(u)
                current_size -= 1
                
                # Try to add a random point
                for _ in range(50):
                    v = rng.choice(all_points)
                    if v in selected_set:
                        continue
                    if try_add(v, selected, dist_from):
                        selected_set.add(v)
                        current_size += 1
                        swapped = True
                        # Don't update best here as size is same, but state changed
                        break
                
                if not swapped:
                    # Re-add u
                    if not try_add(u, selected, dist_from):
                        # Should not happen
                        pass
                    selected_set.add(u)
                    current_size += 1
                else:
                    # Accepted swap, continue to next iteration to try to grow
                    continue

            # 3. Try 2-Swap (Remove 2, Add 2)
            if current_size >= 2:
                idx_u = rng.randint(0, current_size - 1)
                idx_v = rng.randint(0, current_size - 1)
                if idx_u == idx_v:
                    idx_v = (idx_v + 1) % current_size
                    
                u = selected[idx_u]
                v = selected[idx_v]
                
                # Remove u and v
                # Order matters for indices if we remove one then the other
                # Let's remove the higher index first to keep indices valid for the lower one?
                # Or just remove both and rebuild? Rebuild is O(K^2). K is small.
                # Let's use try_remove twice carefully.
                
                # Remove idx_v (if idx_v > idx_u, indices of u don't change. If idx_v < idx_u, u's index changes)
                # Simpler: Copy state, remove, try add, if fail restore.
                
                # To avoid complex index management, we will:
                # 1. Save current state (deep copy)
                # 2. Remove u, v
                # 3. Try add w, x
                # 4. If success, accept. Else restore.
                
                # Deep copy is O(K^2) due to sets.
                # Given K is small, this is acceptable for 2-swaps which are rarer.
                
                selected_backup = list(selected)
                dist_from_backup = [set(s) for s in dist_from]
                selected_set_backup = set(selected_set)
                size_backup = current_size
                
                # Remove u
                try_remove(idx_u, selected, dist_from)
                selected_set.discard(u)
                current_size -= 1
                
                # Adjust idx_v if it was after idx_u
                if idx_v > idx_u:
                    idx_v -= 1
                elif idx_v == idx_u:
                    # Should not happen due to check above
                    pass
                
                # Remove v
                try_remove(idx_v, selected, dist_from)
                selected_set.discard(v)
                current_size -= 1
                
                # Try to add w and x
                swap2_success = False
                w = rng.choice(all_points)
                x = rng.choice(all_points)
                
                # To increase success rate, try a few pairs
                for _ in range(20):
                    w = rng.choice(all_points)
                    x = rng.choice(all_points)
                    if w in selected_set or x in selected_set or w == x:
                        continue
                    
                    # Try add w
                    if try_add(w, selected, dist_from):
                        selected_set.add(w)
                        current_size += 1
                        # Try add x
                        if try_add(x, selected, dist_from):
                            selected_set.add(x)
                            current_size += 1
                            swap2_success = True
                            break
                        else:
                            # Failed to add x, remove w
                            idx_w = selected.index(w)
                            try_remove(idx_w, selected, dist_from)
                            selected_set.discard(w)
                            current_size -= 1
                    # If w failed, try next pair
                
                if swap2_success:
                    # Accepted 2-swap
                    continue
                else:
                    # Restore state
                    selected = selected_backup
                    dist_from = dist_from_backup
                    selected_set = selected_set_backup
                    current_size = size_backup
            
        return local_best, local_best_size

    # Main Loop: Multiple Restarts
    
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        
        # 1. Generate initial set via Randomized Greedy
        selected = []
        dist_from = []
        candidates = list(range(len(all_points)))
        rng.shuffle(candidates)
        
        for idx in candidates:
            p = all_points[idx]
            if try_add(p, selected, dist_from):
                pass # Added
            if time.time() > deadline - 5:
                break
        
        # Now selected is a maximal set (or close to it).
        current_size = len(selected)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(selected)
        
        # 2. Local Search
        if time.time() < deadline - 20:
            # Rebuild dist_from to be safe
            dist_from = rebuild_dist_from(selected)
            
            local_best, local_best_size = local_search(selected, dist_from)
            
            if local_best_size > best_size:
                best_size = local_best_size
                best_set = list(local_best)

    return best_set

# Test/Verify
if __name__ == "__main__":
    res = construct()
    print(len(res))
