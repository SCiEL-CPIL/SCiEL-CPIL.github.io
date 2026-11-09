"""No-isosceles set: find max points in [0,N)^2 with no three forming an isosceles triangle."""
import time
import random
import math

N = 64
TIME_LIMIT = 280  # Safety margin below 330s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(all_points)
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def fast_can_add(p, S, dist_sets):
        """
        O(k) check using precomputed distance sets.
        S: list of points currently in the set
        dist_sets: list where dist_sets[i] is a set of squared distances from S[i] to other points in S
        Returns (valid, distances) where distances[i] = dist2(p, S[i])
        """
        k = len(S)
        if k == 0:
            return True, []
        if k == 1:
            d = dist2(p, S[0])
            return True, [d]
        
        # Calculate distances from p to all points in S
        distances = [0] * k
        for i in range(k):
            distances[i] = dist2(p, S[i])
        
        # Check 1: Apex at existing point S[i].
        # dist(p, S[i]) must not be in dist_sets[i]
        for i in range(k):
            if distances[i] in dist_sets[i]:
                return False, []
        
        # Check 2: Apex at p.
        # No two distances from p to S should be equal.
        if len(set(distances)) < k:
            return False, []
            
        return True, distances

    def greedy_construct(seed_rng):
        """Fast greedy construction with randomized ordering using O(k) validity check."""
        S = []
        dist_sets = []
        candidates = list(range(n_points))
        seed_rng.shuffle(candidates)
        
        for idx in candidates:
            p = all_points[idx]
            valid, distances = fast_can_add(p, S, dist_sets)
            if valid:
                idx_new = len(S)
                S.append(p)
                # Update existing dist_sets
                for i in range(idx_new):
                    dist_sets[i].add(distances[i])
                # Add new dist_set for p
                dist_sets.append(set(distances))
                
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return S, dist_sets

    def local_improvement(pts, dist_sets, seed_rng, max_iters=2000):
        """Try to add more points via local search using fast validity check."""
        S = list(pts)
        ds = [set(s) for s in dist_sets]
        existing_set = set(S)
        remaining = [p for p in all_points if p not in existing_set]
        seed_rng.shuffle(remaining)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(len(remaining)):
                p = remaining[i]
                valid, distances = fast_can_add(p, S, ds)
                if valid:
                    idx_new = len(S)
                    S.append(p)
                    for j in range(idx_new):
                        ds[j].add(distances[j])
                    ds.append(set(distances))
                    existing_set.add(p)
                    remaining.pop(i)
                    improved = True
                    break
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return S, ds

    def swap_improvement(pts, dist_sets, seed_rng, max_iters=500):
        """Try to improve by swapping points using fast validity check."""
        S = list(pts)
        ds = [set(s) for s in dist_sets]
        S_set = set(S)
        current_len = len(S)
        
        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Try to add a random point not in the set
            p_add = seed_rng.choice(all_points)
            if p_add not in S_set:
                valid, distances = fast_can_add(p_add, S, ds)
                if valid:
                    idx_new = len(S)
                    S.append(p_add)
                    for j in range(idx_new):
                        ds[j].add(distances[j])
                    ds.append(set(distances))
                    S_set.add(p_add)
                    current_len = len(S)
                    continue
            
            # Try swap: remove a random point, then try to add a different one
            if current_len > 1:
                idx = seed_rng.randint(0, current_len - 1)
                p_remove = S[idx]
                
                # Rebuild dist_sets without p_remove
                new_S = S[:idx] + S[idx+1:]
                new_ds = []
                for i, pt in enumerate(new_S):
                    dset = set()
                    for j, pt2 in enumerate(new_S):
                        if i != j:
                            dset.add(dist2(pt, pt2))
                    new_ds.append(dset)
                
                # Try to add a new point
                p_new = seed_rng.choice(all_points)
                if p_new not in S_set or p_new == p_remove:
                    valid, distances = fast_can_add(p_new, new_S, new_ds)
                    if valid:
                        idx_new = len(new_S)
                        new_S.append(p_new)
                        for j in range(idx_new):
                            new_ds[j].add(distances[j])
                        new_ds.append(set(distances))
                        S = new_S
                        ds = new_ds
                        S_set = set(S)
                        current_len = len(S)
                        continue
                    else:
                        # Revert: keep original
                        pass
                else:
                    pass
        
        return S, ds

    # Phase 1: Many fast randomized greedy restarts
    num_restarts = 200  # Can afford many more due to O(k) check
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        restart_rng = random.Random(1000 + r * 7919)
        pts, dist_sets = greedy_construct(restart_rng)
        
        if len(pts) > best_len:
            best_pts = pts[:]
            best_dist_sets = [set(s) for s in dist_sets]
            best_len = len(pts)
        
        # Local improvement on this solution if it's competitive
        if len(pts) >= best_len - 2:
            improved_pts, improved_ds = local_improvement(pts, dist_sets, restart_rng, max_iters=500)
            if len(improved_pts) > best_len:
                best_pts = improved_pts[:]
                best_dist_sets = [set(s) for s in improved_ds]
                best_len = len(best_pts)
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 20:
            break
    
    # Phase 2: Final refinement on best
    if time.time() - start_time < TIME_LIMIT - 10 and best_len > 0:
        final_rng = random.Random(99999)
        improved_pts, improved_ds = local_improvement(best_pts, best_dist_sets, final_rng, max_iters=3000)
        if len(improved_pts) > best_len:
            best_pts = improved_pts[:]
            best_dist_sets = [set(s) for s in improved_ds]
            best_len = len(best_pts)
        
        # Swap improvement
        if time.time() - start_time < TIME_LIMIT - 5:
            swap_rng = random.Random(88888)
            swapped_pts, swapped_ds = swap_improvement(best_pts, best_dist_sets, swap_rng, max_iters=1000)
            if len(swapped_pts) > best_len:
                best_pts = swapped_pts[:]
                best_len = len(best_pts)
    
    # Ensure we have at least the baseline
    if not best_pts:
        # Fallback: simple construction
        best_pts = [(0, 0), (1, 0), (0, 1)]
    
    return best_pts
# EVOLVE-BLOCK-END
