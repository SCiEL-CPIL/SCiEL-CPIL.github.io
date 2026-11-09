"""No-isosceles set: find max points in [0,N)^2 with no three forming an isosceles triangle."""
import time
import random
import math

N = 64
TIME_LIMIT = 290  # Safety margin below 330s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    rng = random.Random(42)
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy
    
    def is_isosceles(p, q, r):
        """Check if p, q, r form an isosceles triangle (including degenerate)."""
        d_pq = dist2(p, q)
        d_pr = dist2(p, r)
        d_qr = dist2(q, r)
        # Isosceles if any two sides are equal
        if d_pq == d_pr or d_pq == d_qr or d_pr == d_qr:
            return True
        return False
    
    def can_add(p, pts):
        """Check if point p can be added to pts without creating an isosceles triangle."""
        for i in range(len(pts)):
            pi = pts[i]
            d_p_pi = dist2(p, pi)
            for j in range(i + 1, len(pts)):
                pj = pts[j]
                d_p_pj = dist2(p, pj)
                d_pi_pj = dist2(pi, pj)
                # Check isosceles conditions for triangle (p, pi, pj)
                if d_p_pi == d_p_pj or d_p_pi == d_pi_pj or d_p_pj == d_pi_pj:
                    return False
        return True

    def local_optimize(pts, remaining_indices, seed_rng):
        """
        Greedily add points from remaining_indices to pts if possible.
        Modifies pts and remaining_indices.
        """
        improved = True
        while improved:
            improved = False
            # Shuffle remaining to avoid bias
            seed_rng.shuffle(remaining_indices)
            for idx in remaining_indices:
                p = all_points[idx]
                if can_add(p, pts):
                    pts.append(p)
                    remaining_indices.remove(idx)
                    improved = True
                    if time.time() - start_time > TIME_LIMIT - 5:
                        return pts, remaining_indices
        return pts, remaining_indices

    def greedy_construct(seed_rng):
        """
        Construct a set using greedy approach with random ordering.
        """
        pts = []
        remaining_indices = list(range(len(all_points)))
        seed_rng.shuffle(remaining_indices)
        
        # Fast greedy: just iterate through shuffled list
        for idx in remaining_indices:
            p = all_points[idx]
            if can_add(p, pts):
                pts.append(p)
                remaining_indices.remove(idx) # O(N) removal, acceptable for N=4096 if done rarely? 
                # Actually, removing from list is O(N). 4096 * 100 adds = 400k ops. Fine.
                # But we don't need to maintain remaining_indices perfectly for the greedy phase,
                # we just need to know what's left for local optimization.
                # Let's use a set for remaining for efficiency in local opt.
        # Re-construct remaining set for local optimization
        pt_set = set(pts)
        remaining_indices = [i for i in range(len(all_points)) if all_points[i] not in pt_set]
        
        return pts, remaining_indices

    # Main Search Loop: Iterated Local Search (ILS)
    # We will perform many iterations of:
    # 1. Greedy start
    # 2. Local Optimize
    # 3. If better, keep.
    # 4. Perturb (remove k points) and Local Optimize again.
    
    num_iterations = 0
    max_iterations = 50000 # Cap to prevent infinite loop if time check fails
    
    while time.time() - start_time < TIME_LIMIT - 10 and num_iterations < max_iterations:
        num_iterations += 1
        iter_rng = random.Random(num_iterations * 101 + 7)
        
        # Phase 1: Greedy Construction
        pts, remaining = greedy_construct(iter_rng)
        
        # Phase 2: Local Optimization
        pts, remaining = local_optimize(pts, remaining, iter_rng)
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = list(pts)
        
        # Phase 3: Perturbation and Re-optimization
        # If we have a decent set, try to escape local optima
        if len(pts) > 5:
            # Try different perturbation strengths
            for k in [1, 2, 3, 5]:
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                # Create a copy to perturb
                current_pts = list(pts)
                current_remaining = list(remaining)
                
                # Remove k random points from current_pts
                if len(current_pts) <= k:
                    continue
                    
                removed = []
                for _ in range(k):
                    if not current_pts:
                        break
                    idx_to_remove = iter_rng.randint(0, len(current_pts) - 1)
                    p_removed = current_pts.pop(idx_to_remove)
                    removed.append(p_removed)
                
                # Add removed points back to remaining
                current_remaining.extend(removed)
                
                # Re-optimize
                current_pts, current_remaining = local_optimize(current_pts, current_remaining, iter_rng)
                
                if len(current_pts) > best_len:
                    best_len = len(current_pts)
                    best_pts = list(current_pts)
    
    return [(x, y) for x, y in best_pts]
# EVOLVE-BLOCK-END
