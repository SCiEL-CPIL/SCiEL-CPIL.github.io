def construct():
    import time
    import random
    
    deadline = time.time() + 290
    
    N = 64
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_isosceles(p1, p2, p3):
        d12 = sq_dist(p1, p2)
        d13 = sq_dist(p1, p3)
        d23 = sq_dist(p2, p3)
        return d12 == d13 or d12 == d23 or d13 == d23

    def can_add_fast(p, S, dist_sets):
        # dist_sets: dict point -> set of squared distances to other points in S
        for q in S:
            d = sq_dist(p, q)
            if d in dist_sets[q]:
                return False
            # Check if p has duplicate distance to existing points
            # This part is tricky to do fast without a set of p's distances
            # We can just collect them
        # We need to check if p has duplicate distances to S
        # And if any q in S has d(p,q) in dist_sets[q]
        
        dists_from_p = []
        for q in S:
            d = sq_dist(p, q)
            if d in dists_from_p:
                return False
            dists_from_p.append(d)
            if d in dist_sets[q]:
                return False
        return True

    def add_point(S, dist_sets, p):
        S.append(p)
        dists = []
        for q in S[:-1]:
            d = sq_dist(p, q)
            dists.append(d)
            dist_sets[q].add(d)
        dist_sets[p] = set(dists)

    def remove_point(S, dist_sets, p):
        S.remove(p)
        for q in S:
            d = sq_dist(p, q)
            if d in dist_sets[q]:
                dist_sets[q].discard(d)
        del dist_sets[p]

    def greedy(order):
        S = []
        dist_sets = {}
        for p in order:
            if time.time() > deadline:
                break
            if p in dist_sets:
                continue
            if can_add_fast(p, S, dist_sets):
                add_point(S, dist_sets, p)
        return S, dist_sets

    def local_search(S, dist_sets):
        # Try to swap points
        current_S = list(S)
        current_dist = {k: v.copy() for k, v in dist_sets.items()}
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing a random point and adding a random candidate
            if not current_S:
                break
            idx = random.randint(0, len(current_S) - 1)
            p_remove = current_S[idx]
            
            # Remove p_remove
            remove_point(current_S, current_dist, p_remove)
            
            # Find candidates not in current_S
            candidates = [p for p in all_points if p not in current_dist]
            random.shuffle(candidates)
            
            added = False
            for p_add in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add_fast(p_add, current_S, current_dist):
                    add_point(current_S, current_dist, p_add)
                    added = True
                    break # Accept first improvement? Or try more?
            
            if added and len(current_S) > len(S):
                S = current_S
                dist_sets = current_dist
                improved = True
            else:
                # Revert? 
                # We modified current_S and current_dist.
                # We should keep the best found.
                # If we didn't improve, we should revert to the start of the while loop state?
                # Actually, we just try one swap. If it fails, we try another.
                # But we mutated current_S.
                # Better to work on copies.
                pass
                
        return S, dist_sets

    best_S = []
    best_dist = {}
    
    rng = random.Random(42)
    
    # Multiple Greedy Runs
    for i in range(20):
        if time.time() > deadline - 10:
            break
        order = all_points.copy()
        rng.shuffle(order)
        S, dist_sets = greedy(order)
        if len(S) > len(best_S):
            best_S = S
            best_dist = dist_sets
            # Optional: Local Search
            # S2, dist2 = local_search(best_S, best_dist)
            # if len(S2) > len(best_S):
            #     best_S = S2
            #     best_dist = dist2

    # Final Validation and Pruning
    # Check all triples in best_S
    # If isosceles, remove one point
    # This is O(K^3). K ~ 100. 10^6 ops. OK.
    valid_S = list(best_S)
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        n = len(valid_S)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_isosceles(valid_S[i], valid_S[j], valid_S[k]):
                        # Remove valid_S[k]
                        valid_S.pop(k)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    return valid_S
