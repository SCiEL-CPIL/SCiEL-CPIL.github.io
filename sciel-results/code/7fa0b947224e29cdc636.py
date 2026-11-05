import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each index
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            
            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def build_initial_cover():
        """
        Construct a valid cover using a greedy approach.
        """
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Randomized greedy with restarts to find a good initial solution
        best_selected = None
        best_len = float('inf')
        
        # We can try a few different random orders or strategies
        for seed in [42, 123, 999]:
            rng = np.random.default_rng(seed)
            is_covered = np.zeros(total_points, dtype=bool)
            selected = []
            
            # Pre-compute a random priority for each point to break ties or guide selection?
            # Standard greedy: pick the ball that covers the most uncovered points.
            # To speed up, we can just iterate.
            
            while not np.all(is_covered):
                # Find the ball covering the most uncovered points
                # Vectorized: counts = sum(~is_covered[neighbors], axis=1)
                uncovered = ~is_covered
                counts = np.sum(uncovered[neighbors], axis=1)
                max_count = np.max(counts)
                
                if max_count == 0:
                    break
                
                # Pick randomly among those with max count to add diversity
                candidates = np.where(counts == max_count)[0]
                best_idx = candidates[rng.integers(len(candidates))]
                
                selected.append(best_idx)
                is_covered[neighbors[best_idx]] = True
            
            if len(selected) < best_len:
                best_len = len(selected)
                best_selected = selected[:]
                
        return best_selected

    def prune(selected_indices):
        """
        Remove redundant codewords.
        """
        if not selected_indices:
            return []
            
        selected_set = set(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_set:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of the list to allow removal
            for idx in list(selected_set):
                # Check if all points in this ball have coverage > 1
                if np.all(coverage_count[neighbors[idx]] > 1):
                    # It is redundant
                    coverage_count[neighbors[idx]] -= 1
                    selected_set.discard(idx)
                    changed = True
        
        return list(selected_set)

    def local_search_optimize(selected_indices, deadline):
        """
        Local search using 1-move (swap) and 2-move (remove 2, add 1) operations.
        """
        selected_list = prune(selected_indices)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improvement_made = False
            
            # Strategy 1: Try to remove a single redundant codeword
            # We can just run prune again, but that's O(N^2) worst case.
            # Instead, check if any codeword is redundant.
            # To speed up, we can just call prune, it's fast enough for ~500 items.
            pruned = prune(selected_list)
            if len(pruned) < len(selected_list):
                selected_list = pruned
                selected_set = set(selected_list)
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[neighbors[idx]] += 1
                improvement_made = True
                no_improve_count = 0
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                continue

            # Strategy 2: 2-opt (Remove 2, Add 1)
            # We look for c1, c2 such that R1 U R2 can be covered by c_new
            # R1 = points in B(c1) with count 1 (covered only by c1)
            # R2 = points in B(c2) with count 1 (covered only by c2)
            
            # Sample a subset of codewords to try as c1 to save time
            # If the list is large, sampling is good. If small, try all.
            sample_size = min(len(selected_list), 20)
            c1_indices = rng.choice(len(selected_list), sample_size, replace=False)
            
            found_improvement = False
            for i in c1_indices:
                if found_improvement:
                    break
                c1 = selected_list[i]
                ball1 = neighbors[c1]
                
                # Find R1: points in ball1 with coverage_count == 1
                r1_mask = coverage_count[ball1] == 1
                r1_points = ball1[r1_mask]
                
                if len(r1_points) == 0:
                    # c1 is redundant, prune will catch it next loop
                    continue
                    
                if len(r1_points) > 3:
                    # Unlikely to be coverable by one other codeword's R2 union
                    # Actually, R1 U R2 must be in a ball. Max ball size 17.
                    # If R1 is large, it's hard.
                    pass

                # Try other codewords as c2
                # We can sample c2s or iterate all. Iterating all is ~500 * 17 checks.
                for j in range(len(selected_list)):
                    if j == i:
                        continue
                    c2 = selected_list[j]
                    ball2 = neighbors[c2]
                    
                    # Find R2
                    r2_mask = coverage_count[ball2] == 1
                    r2_points = ball2[r2_mask]
                    
                    # Candidate set for c_new must cover R1 and R2
                    # Intersect neighbors of all points in R1 U R2
                    # To be efficient, start with neighbors of points in R1
                    # Then intersect with neighbors of points in R2
                    
                    # Optimization: If R1 and R2 are disjoint and small, 
                    # the intersection of their neighbor sets is the candidate pool.
                    
                    # Get candidate set from R1
                    cand_set = None
                    for p in r1_points:
                        ns = set(neighbors[p].tolist())
                        cand_set = ns if cand_set is None else (cand_set & ns)
                        if not cand_set:
                            break
                            
                    if not cand_set:
                        continue
                        
                    # Intersect with R2
                    for p in r2_points:
                        ns = set(neighbors[p].tolist())
                        cand_set = cand_set & ns
                        if not cand_set:
                            break
                            
                    if not cand_set:
                        continue
                        
                    # Check if any candidate is valid
                    # A valid candidate c_new must:
                    # 1. Not be c1 or c2 (usually implied if it covers R1 which is unique to c1, 
                    #    unless c_new == c1, but then R2 must be in B(c1), which implies R2 points 
                    #    are covered by c1, so they wouldn't be in R2 (count 1 for c2). 
                    #    Wait, if p in R2, count[p]=1 and p in B(c2). If p in B(c1), count[p]>=2. 
                    #    So R1 and R2 are disjoint and no point in R1 is in B(c2) and vice versa?
                    #    Not necessarily. 
                    #    If p in R1, count[p]=1, p in B(c1). So p is NOT in B(c2) for any c2 != c1.
                    #    So R1 is disjoint from B(c2) for all c2 != c1.
                    #    Similarly R2 is disjoint from B(c1).
                    #    So R1 U R2 is disjoint from B(c1) and B(c2).
                    #    Thus c_new cannot be c1 or c2.
                    # 2. Be not in selected_set (or if it is, we are just moving weight? 
                    #    No, we are removing c1, c2 and adding c_new. 
                    #    If c_new is already in selected_set, we are effectively moving c1, c2 to c_new? 
                    #    No, c_new is a specific index. If it's already selected, we just remove c1, c2.
                    #    But c_new must cover R1 U R2. 
                    #    If c_new is already selected, does it cover R1? 
                    #    R1 points have count 1 (from c1). If c_new is selected and covers R1, 
                    #    then count[R1] would be >= 2. Contradiction. 
                    #    So c_new cannot be in selected_set.
                    
                    for c_new in cand_set:
                        if c_new in selected_set:
                            continue
                        
                        # Perform the move: remove c1, c2, add c_new
                        # Update coverage counts
                        coverage_count[ball1] -= 1
                        coverage_count[ball2] -= 1
                        coverage_count[neighbors[c_new]] += 1
                        
                        # Update lists
                        # Remove c1, c2
                        # We need to remove by value.
                        # Using list.remove is O(N). 
                        # For N=500, this is fine.
                        
                        # Check if the move is valid (it should be, by construction)
                        # But let's verify no point has count 0
                        # We only need to check points in B(c1) U B(c2)
                        # Points in R1 had count 1, now 0 (from c1) + 1 (from c_new) = 1. OK.
                        # Points in R2 had count 1, now 0 (from c2) + 1 (from c_new) = 1. OK.
                        # Other points in B(c1) had count >= 2. Now count >= 1. OK.
                        # Other points in B(c2) had count >= 2. Now count >= 1. OK.
                        # Points outside B(c1) U B(c2) are unchanged.
                        
                        # So the move is always valid if c_new covers R1 U R2.
                        
                        # Execute
                        selected_list.remove(c1)
                        selected_list.remove(c2)
                        selected_list.append(c_new)
                        
                        selected_set.discard(c1)
                        selected_set.discard(c2)
                        selected_set.add(c_new)
                        
                        improvement_made = True
                        found_improvement = True
                        no_improve_count = 0
                        
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        
                        break # Break c_new loop
                    if found_improvement:
                        break # Break c2 loop
                
            if not improvement_made:
                no_improve_count += 1
                
                # If stuck, try a random perturbation? 
                # Or just rely on the 2-opt. 
                # If no_improve_count gets high, we might be in a local minimum.
                # A simple random swap might help escape.
                if no_improve_count % 100 == 0:
                    # Random swap
                    if len(selected_list) > 0:
                        c = selected_list[rng.integers(len(selected_list))]
                        c_prime = neighbors[c][rng.integers(1, ball_size)]
                        if c_prime not in selected_set:
                            # Tentative swap
                            coverage_count[neighbors[c]] -= 1
                            coverage_count[neighbors[c_prime]] += 1
                            
                            # Check validity
                            if not np.any(coverage_count[neighbors[c]] == 0):
                                selected_list[selected_list.index(c)] = c_prime
                                selected_set.discard(c)
                                selected_set.add(c_prime)
                                # Don't update best_len unless it improves, but this is a move, size is same.
                                # However, it might enable future deletions.
                                # We don't reset no_improve_count on neutral moves, 
                                # but we do break the "stuck" state.
                            else:
                                # Revert
                                coverage_count[neighbors[c_prime]] -= 1
                                coverage_count[neighbors[c]] += 1
        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Build initial solution
    best_code = build_initial_cover()
    best_code = prune(best_code)
    
    # Local Search
    best_code = local_search_optimize(best_code, deadline)
    
    # Final prune
    best_code = prune(best_code)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
