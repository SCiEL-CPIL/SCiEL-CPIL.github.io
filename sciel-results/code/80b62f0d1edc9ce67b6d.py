import numpy as np
import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    all_points_arr = np.array(all_points, dtype=np.int32)
    
    def get_distance_matrix(points):
        if len(points) == 0:
            return np.array([]).reshape(0, 0)
        P = np.array(points, dtype=np.int32)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        return np.sum(diff**2, axis=2)

    def count_violations(points, D):
        """
        Counts the number of isosceles triangles in the set.
        Returns 0 if no isosceles triangles exist.
        """
        k = len(points)
        if k < 3:
            return 0
        
        # D is k x k matrix of squared distances.
        # For each pair (i, j), check if there exists k such that D[i,k] == D[i,j] or D[i,k] == D[j,k] ... 
        # Actually, simpler:
        # For each point i, count how many pairs (j, k) form an isosceles triangle with apex i.
        # This means D[i,j] == D[i,k].
        # Also need to handle cases where apex is j or k.
        # A triangle (i, j, k) is isosceles if:
        # 1. D[i,j] == D[i,k] (Apex i)
        # 2. D[i,j] == D[j,k] (Apex j)
        # 3. D[i,k] == D[j,k] (Apex k)
        
        # Note: Equilateral triangles satisfy all 3 conditions.
        # We want to count the number of such triangles.
        # If we sum for each i, the number of pairs (j,k) with D[i,j]==D[i,k],
        # we count each equilateral triangle 3 times.
        # We count each non-equilateral isosceles triangle exactly once (at its unique apex).
        
        total_violations = 0
        for i in range(k):
            # Get distances from i to all other points
            dists = D[i, :].copy()
            dists[i] = -1 # Exclude self
            
            # Count pairs (j, k) such that dists[j] == dists[k]
            # We can use unique counts
            vals, counts = np.unique(dists, return_counts=True)
            for c in counts:
                if c >= 2:
                    # Number of pairs is c * (c-1) / 2
                    total_violations += c * (c - 1) // 2
        
        # This counts equilateral triangles 3 times.
        # The task says "no three points forming an isosceles triangle".
        # So any isosceles triangle is a violation.
        # We just need the count to be 0 for a valid set.
        # For optimization, we want to minimize this count.
        # The factor of 3 for equilateral triangles doesn't matter for the "is it 0?" check,
        # but for gradient-like steps, it's a consistent metric.
        
        return total_violations

    # Greedy Construction
    points = []
    point_set = set()
    D = np.array([]).reshape(0, 0)
    
    # Start with a random seed to avoid bias, or start empty.
    # Let's start with a few random points that are far apart?
    # Just greedy add: pick point with min violations.
    
    # To speed up greedy, we can precompute distances between all grid points?
    # 64x64 = 4096 points. 4096^2 = 16 million pairs. 
    # Storing 16M ints is ~64MB. Doable.
    # But calculating violations for each candidate is O(K^2) if done naively.
    # With precomputed D_grid, adding a point p to set S:
    # Violations involving p:
    # 1. Apex p: pairs (a,b) in S such that d(p,a)==d(p,b).
    # 2. Apex a: pairs (p,b) in S such that d(a,p)==d(a,b).
    # 3. Apex b: pairs (a,p) in S such that d(b,p)==d(b,a).
    
    # Let's use the incremental check with precomputed grid distances.
    
    # Precompute squared distances between all grid points
    # This is a 4096x4096 matrix.
    print("Precomputing grid distances...", flush=True)
    D_grid = get_distance_matrix(all_points)
    print("Done.", flush=True)
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # Greedy Phase
    # Start with 0 points.
    # At each step, evaluate all candidates not in set.
    # This is O(N_grid * K^2) per step? No.
    # For a candidate p, we need to check against existing points S.
    # We can use the precomputed D_grid.
    
    # Let's try a simpler heuristic first: Random restart + Local Search.
    # The greedy "min violations" is slow if done naively for 4096 candidates.
    # But 4096 * (small K) is fast.
    
    best_points = []
    best_violations = float('inf')
    
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Random start
        current_points = [random.choice(all_points) for _ in range(5)]
        current_set = set(current_points)
        
        # Ensure no duplicates in start
        while len(current_set) < len(current_points):
             current_points.append(random.choice(all_points))
             current_set.add(tuple(current_points[-1])) # Ensure tuple for set
        current_points = list(current_set)
        
        D_cur = get_distance_matrix(current_points)
        cur_viol = count_violations(current_points, D_cur)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add a point
            # Sample some random candidates to speed up
            candidates = random.sample([p for p in all_points if p not in current_set], min(50, N*N - len(current_set)))
            
            best_candidate = None
            best_candidate_viol = cur_viol
            
            for p in candidates:
                # Calculate new violations
                # We need D_new.
                # Instead of building full D_new, we can compute delta.
                # But count_violations is simple enough to just build D_new for small K?
                # K is likely < 100. 100x100 matrix is trivial.
                
                new_points = current_points + [p]
                D_new = get_distance_matrix(new_points)
                new_viol = count_violations(new_points, D_new)
                
                if new_viol < best_candidate_viol:
                    best_candidate_viol = new_viol
                    best_candidate = p
            
            if best_candidate is not None:
                current_points.append(best_candidate)
                current_set.add(best_candidate)
                cur_viol = best_candidate_viol
                improved = True
                if cur_viol == 0:
                    # Try to add more if 0 violations
                    continue
            else:
                # Try to swap
                if len(current_points) > 5:
                    idx = random.randint(0, len(current_points) - 1)
                    old_p = current_points[idx]
                    candidates = random.sample([p for p in all_points if p not in current_set], min(50, N*N - len(current_set)))
                    
                    best_candidate = None
                    best_candidate_viol = cur_viol
                    
                    for p in candidates:
                        # Swap old_p with p
                        new_points = current_points[:idx] + [p] + current_points[idx+1:]
                        D_new = get_distance_matrix(new_points)
                        new_viol = count_violations(new_points, D_new)
                        
                        if new_viol < best_candidate_viol:
                            best_candidate_viol = new_viol
                            best_candidate = p
                    
                    if best_candidate is not None:
                        current_points[idx] = best_candidate
                        current_set.remove(old_p)
                        current_set.add(best_candidate)
                        cur_viol = best_candidate_viol
                        improved = True

        # Update best
        if cur_viol < best_violations or (cur_viol == best_violations and len(current_points) > len(best_points)):
            best_violations = cur_viol
            best_points = current_points[:]
            
    # If best_violations > 0, we have an invalid set. 
    # The task requires "no three points forming an isosceles triangle".
    # If we can't find a valid set, we should return the largest valid set found?
    # The score is "number of points". 
    # If the set is invalid, does it score 0? 
    # "Score: number of points". 
    # "No-isosceles set task... with no three points forming an isosceles triangle."
    # Usually, if the constraint is violated, the score is 0 or the constraint is hard.
    # Given "invalid result... scores 0", I assume returning an invalid set scores 0.
    # So we must ensure we return a valid set.
    
    if best_violations > 0:
        # Fallback: Find a small valid set (e.g., 2 points are always valid)
        # Or prune the best set to make it valid?
        # Pruning is hard. 
        # Let's just return the first 2 points from all_points, which is always valid.
        # Or try to find a valid set via greedy from scratch with a stricter rule.
        
        # Greedy valid set construction:
        valid_points = []
        valid_set = set()
        for p in all_points:
            # Check if p can be added to valid_points
            new_points = valid_points + [p]
            if len(new_points) < 3:
                valid_points = new_points
                valid_set.add(p)
                continue
            
            D_new = get_distance_matrix(new_points)
            v = count_violations(new_points, D_new)
            if v == 0:
                valid_points = new_points
                valid_set.add(p)
                
        if len(valid_points) > len(best_points) or best_violations > 0:
            best_points = valid_points
            best_violations = 0 # It is valid

    return [(int(x), int(y)) for x, y in best_points]

# EVOLVE-BLOCK-END
