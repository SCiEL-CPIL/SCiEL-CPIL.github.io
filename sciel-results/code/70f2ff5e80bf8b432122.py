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
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy
    
    def can_add(p, pts):
        """Check if point p can be added to pts without creating an isosceles triangle."""
        k = len(pts)
        if k == 0:
            return True
        if k == 1:
            return True
        
        # Precompute distances from p to all existing points
        dp = [dist2(p, pts[i]) for i in range(k)]
        
        # Check p as apex: all distances from p must be distinct
        if len(set(dp)) != k:
            return False
        
        # For each existing point A_i as apex: dist(A_i, p) must not equal dist(A_i, A_j) for j != i
        for i in range(k):
            di = dp[i]
            Ai = pts[i]
            for j in range(k):
                if j == i:
                    continue
                if dist2(Ai, pts[j]) == di:
                    return False
        
        return True
    
    def greedy_construct(seed_rng):
        """Greedy construction with randomized ordering."""
        pts = []
        candidates = list(range(len(all_points)))
        seed_rng.shuffle(candidates)
        
        for idx in candidates:
            p = all_points[idx]
            if can_add(p, pts):
                pts.append(p)
            # Time check
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return pts
    
    def local_improvement(pts, seed_rng, max_iters=1000):
        """Try to add more points via local search by retrying candidates."""
        pts = list(pts)
        existing_set = set(pts)
        remaining = [p for p in all_points if p not in existing_set]
        seed_rng.shuffle(remaining)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(len(remaining)):
                p = remaining[i]
                if can_add(p, pts):
                    pts.append(p)
                    existing_set.add(p)
                    remaining.pop(i)
                    improved = True
                    break
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return pts
    
    def swap_improvement(pts, seed_rng, max_iters=200):
        """Try to improve by swapping points: remove one, try to add another."""
        pts = list(pts)
        pts_set = set(pts)
        current_len = len(pts)
        
        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Try to add a random point not in the set
            p_add = seed_rng.choice(all_points)
            if p_add not in pts_set:
                if can_add(p_add, pts):
                    pts.append(p_add)
                    pts_set.add(p_add)
                    current_len = len(pts)
                    continue
            
            # Try swap: remove a random point, then try to add a different one
            if current_len > 1:
                idx = seed_rng.randint(0, current_len - 1)
                p_remove = pts[idx]
                pts.pop(idx)
                pts_set.remove(p_remove)
                
                # Try to add a new point
                p_new = seed_rng.choice(all_points)
                if p_new not in pts_set:
                    if can_add(p_new, pts):
                        pts.append(p_new)
                        pts_set.add(p_new)
                    else:
                        # Revert
                        pts.append(p_remove)
                        pts_set.add(p_remove)
                else:
                    # Revert
                    pts.append(p_remove)
                    pts_set.add(p_remove)
        
        return pts
    
    # Multiple randomized greedy restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        restart_rng = random.Random(1000 + r * 7919)
        pts = greedy_construct(restart_rng)
        
        if len(pts) > best_len:
            best_pts = pts[:]
            best_len = len(pts)
        
        # Local improvement on this solution
        if len(pts) >= best_len - 1:
            improved = local_improvement(pts, restart_rng)
            if len(improved) > best_len:
                best_pts = improved[:]
                best_len = len(best_pts)
            
            # Swap improvement on the best
            if time.time() - start_time < TIME_LIMIT - 20:
                swap_rng = random.Random(2000 + r * 104729)
                swapped = swap_improvement(best_pts, swap_rng)
                if len(swapped) > best_len:
                    best_pts = swapped[:]
                    best_len = len(best_pts)
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 15:
            break
    
    # Final local improvement pass on best
    if time.time() - start_time < TIME_LIMIT - 10 and best_len > 0:
        final_rng = random.Random(99999)
        improved = local_improvement(best_pts, final_rng, max_iters=2000)
        if len(improved) > best_len:
            best_pts = improved[:]
            best_len = len(best_pts)
        
        # Final swap improvement
        if time.time() - start_time < TIME_LIMIT - 5:
            swap_rng = random.Random(88888)
            swapped = swap_improvement(best_pts, swap_rng, max_iters=500)
            if len(swapped) > best_len:
                best_pts = swapped[:]
                best_len = len(best_pts)
    
    # Ensure we have at least the baseline
    if not best_pts:
        # Fallback: simple 1D Sidon-like construction
        xs, used = [], set()
        x = 0
        while x < N:
            diffs = {abs(x - y) for y in xs}
            if not (diffs & used):
                used |= diffs
                xs.append(x)
            x += 1
        best_pts = [(x, 0) for x in xs]
    
    return best_pts
# EVOLVE-BLOCK-END
