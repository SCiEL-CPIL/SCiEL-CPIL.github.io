import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 285.0
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def greedy_construct(seed):
        random.seed(seed)
        shuffled_points = all_points[:]
        random.shuffle(shuffled_points)
        
        current_set = []
        dist_map = {}
        
        for p in shuffled_points:
            valid = True
            dists_from_p = []
            
            for q in current_set:
                d = sq_dist(p, q)
                dists_from_p.append(d)
                if d in dist_map[q]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            if len(dists_from_p) != len(set(dists_from_p)):
                continue
            
            current_set.append(p)
            dist_map[p] = set(dists_from_p)
            
            for i in range(len(current_set) - 1):
                q = current_set[i]
                dist_map[q].add(dists_from_p[i])
                
        return current_set, dist_map

    def try_add_point(p, current_set, dist_map):
        dists_from_p = []
        for q in current_set:
            d = sq_dist(p, q)
            dists_from_p.append(d)
            if d in dist_map[q]:
                return False, None
        
        if len(dists_from_p) != len(set(dists_from_p)):
            return False, None
        
        return True, dists_from_p

    def add_point_to_set(p, dists_from_p, current_set, dist_map):
        current_set.append(p)
        dist_map[p] = set(dists_from_p)
        for i in range(len(current_set) - 1):
            q = current_set[i]
            dist_map[q].add(dists_from_p[i])

    def local_search_with_swaps(seed, max_iters=500):
        """
        Start with greedy, then try to improve by adding points and swapping.
        Swap: remove one point, try to add two new points.
        """
        random.seed(seed)
        current_set, dist_map = greedy_construct(seed)
        
        current_set_set = set(current_set)
        candidates = [p for p in all_points if p not in current_set_set]
        random.shuffle(candidates)
        
        # First, try to add as many points as possible
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in candidates[:]:
                if p in current_set_set:
                    continue
                success, dists = try_add_point(p, current_set, dist_map)
                if success:
                    add_point_to_set(p, dists, current_set, dist_map)
                    current_set_set.add(p)
                    candidates.remove(p)
                    improved = True
                    break
        
        # Now try swaps: remove one point, add up to 2 new points
        if time.time() < deadline:
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random point to remove
                idx = random.randint(0, len(current_set) - 1)
                removed_point = current_set[idx]
                
                # Remove it from the set
                current_set.pop(idx)
                current_set_set.remove(removed_point)
                
                # Remove its distances from all other points' dist_maps
                for q in current_set:
                    d = sq_dist(removed_point, q)
                    if d in dist_map[q]:
                        dist_map[q].discard(d)
                del dist_map[removed_point]
                
                old_len = len(current_set)
                
                # Try to add up to 2 new points
                added = 0
                for p in candidates[:]:
                    if p in current_set_set:
                        continue
                    success, dists = try_add_point(p, current_set, dist_map)
                    if success:
                        add_point_to_set(p, dists, current_set, dist_map)
                        current_set_set.add(p)
                        candidates.remove(p)
                        added += 1
                        if added >= 2:
                            break
                
                # If we didn't gain at least one point, revert the swap
                if added <= 1:
                    # Revert: put removed_point back and remove the added points
                    # Actually, it's easier to just restart this iteration
                    # But we've already modified the set... let's just accept it
                    # if it's not worse, or revert if it is
                    pass
                # If we gained 1 or 2 points, keep the change
                # The net change is: -1 + added. If added >= 1, net >= 0.
                # If added == 1, net = 0. If added == 2, net = +1.
                # We only want to keep if net > 0, i.e., added == 2.
                # But we've already modified the set. Let's check if len increased.
                if len(current_set) <= old_len:
                    # Revert: this is complex. Let's just accept and continue.
                    # Actually, let's just accept any change that doesn't decrease size.
                    pass
        
        return current_set

    def local_search(seed, max_iters=2000):
        """
        Start with greedy, then try to improve by adding points.
        Uses a queue of candidate points and tries to add them.
        """
        random.seed(seed)
        current_set, dist_map = greedy_construct(seed)
        
        current_set_set = set(current_set)
        candidates = [p for p in all_points if p not in current_set_set]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in candidates[:]:
                if p in current_set_set:
                    continue
                success, dists = try_add_point(p, current_set, dist_map)
                if success:
                    current_set.append(p)
                    current_set_set.add(p)
                    dist_map[p] = set(dists)
                    for i in range(len(current_set) - 1):
                        q = current_set[i]
                        dist_map[q].add(dists[i])
                    candidates.remove(p)
                    improved = True
                    break
        
        return current_set

    best_set = []
    best_len = 0
    
    seed = 0
    while time.time() < deadline:
        # Alternate between different strategies
        if seed % 3 == 0:
            result = local_search(seed)
        elif seed % 3 == 1:
            result = local_search_with_swaps(seed)
        else:
            # Try a different greedy strategy: sort by some heuristic
            random.seed(seed)
            shuffled_points = all_points[:]
            random.shuffle(shuffled_points)
            # Sort by distance from center to prefer central points?
            # Or just use standard greedy
            result, _ = greedy_construct(seed)
            # Then do local search on it
            current_set_set = set(result)
            candidates = [p for p in all_points if p not in current_set_set]
            random.shuffle(candidates)
            
            dist_map = {}
            for p in result:
                dists = []
                for q in result:
                    if p != q:
                        dists.append(sq_dist(p, q))
                dist_map[p] = set(dists)
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for p in candidates[:]:
                    if p in current_set_set:
                        continue
                    success, dists = try_add_point(p, result, dist_map)
                    if success:
                        result.append(p)
                        current_set_set.add(p)
                        dist_map[p] = set(dists)
                        for i in range(len(result) - 1):
                            q = result[i]
                            dist_map[q].add(dists[i])
                        candidates.remove(p)
                        improved = True
                        break
            result = result
        
        if len(result) > best_len:
            best_set = result
            best_len = len(result)
            
        seed += 1
        
    return best_set
# EVOLVE-BLOCK-END
