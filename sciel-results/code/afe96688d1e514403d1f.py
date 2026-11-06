import itertools
import time
import random
import math

def _is_sidon(subset):
    """Check if a subset has all pair sums (including diagonals) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_subset_size(S):
    """Find the largest Sidon subset of S using backtracking with pruning."""
    S = sorted(S)
    n = len(S)
    
    # Quick upper bound: if all pair sums are unique, it's Sidon
    if _is_sidon(S):
        return n
    
    # Try sizes from n-1 down to 1
    for size in range(n - 1, 0, -1):
        # Use backtracking with early termination
        def backtrack(start, current, current_sums):
            if len(current) == size:
                return True
            remaining_needed = size - len(current)
            remaining_available = n - start
            if remaining_available < remaining_needed:
                return False
            
            for i in range(start, n):
                val = S[i]
                # Pruning: check if adding S[i] to current keeps it Sidon
                # We check if val + x creates a duplicate sum for any x in current
                # Also check if val + val creates a duplicate
                is_valid = True
                new_sums = set(current_sums)
                
                # Check val + val
                if val + val in new_sums:
                    is_valid = False
                else:
                    new_sums.add(val + val)
                    
                # Check val + x for x in current
                if is_valid:
                    for x in current:
                        s = val + x
                        if s in new_sums:
                            is_valid = False
                            break
                        new_sums.add(s)
                
                if is_valid:
                    if backtrack(i + 1, current + [val], new_sums):
                        return True
            return False
        
        # Initial sums set for empty current is empty
        if backtrack(0, [], set()):
            return size
    
    return 1

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    
    # Simulated Annealing to maximize sum collisions (minimize Sidon subset size)
    # State: a set of 12 distinct integers in [0, 128]
    
    def get_score(S):
        """Returns the size of the largest Sidon subset. Lower is better."""
        return max_sidon_subset_size(S)
    
    def collision_density(S):
        """
        Heuristic score for SA: count how many pair sums are shared.
        Higher density -> more collisions -> likely smaller Sidon subsets.
        This is much faster than exact max_sidon_subset_size.
        """
        sums = {}
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    sums[s] += 1
                else:
                    sums[s] = 1
        # Sum of (count-1) for all sums, representing number of "extra" collisions
        # Or simply sum of squares of counts?
        # Let's use sum of (c choose 2) which counts pairs of pairs that collide.
        # This is a good proxy for "hardness".
        total_collisions = 0
        for c in sums.values():
            if c > 1:
                total_collisions += c * (c - 1) // 2
        return total_collisions

    # Initialize with a random set
    rng = random.Random(42)
    current_set = rng.sample(range(129), 12)
    current_score = get_score(current_set)
    current_density = collision_density(current_set)
    
    best_set = list(current_set)
    best_score = current_score
    
    # SA Parameters
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    while T > min_T and time.time() < deadline:
        # Perturbation: pick a random element and change it to a random other value
        idx = rng.randint(0, 11)
        old_val = current_set[idx]
        new_val = rng.randint(0, 128)
        
        if new_val == old_val:
            continue
            
        # Check if new_val is already in the set (excluding the one we're replacing)
        if any(v == new_val for i, v in enumerate(current_set) if i != idx):
            continue

        # Create candidate
        candidate_set = list(current_set)
        candidate_set[idx] = new_val
        
        # Evaluate candidate
        # First, quick check with density? Or just exact score?
        # Exact score is expensive. Let's use density as a surrogate for the SA step,
        # but we must track the best exact score separately.
        
        candidate_density = collision_density(candidate_set)
        delta_density = candidate_density - current_density
        
        # We want to MAXIMIZE density (minimize Sidon size).
        # So if delta_density > 0, it's a "better" move in terms of collisions.
        # In SA, we accept if it's better, or with probability exp(-delta/T) if worse.
        # Here "worse" means lower density (fewer collisions).
        # So if delta_density < 0, we accept with prob exp(delta_density / T).
        # Wait, standard SA minimizes energy. Here we maximize density.
        # Let Energy = -Density.
        # delta_E = -delta_density.
        # Accept if delta_E < 0 (density increases) OR with prob exp(-delta_E/T) = exp(delta_density/T).
        
        if delta_density > 0:
            accept = True
        else:
            # delta_density is negative
            accept = rng.random() < math.exp(delta_density / T)
            
        if accept:
            current_set = candidate_set
            current_density = candidate_density
            
            # Periodically check exact score to update best
            # Checking every step is too slow. Check every N steps or if density improves significantly?
            # Let's check every 100 steps or if T drops below a threshold.
            # Actually, let's just check if it's likely better.
            # Or just check every 50 steps.
            if rng.randint(0, 49) == 0 or T < 0.1:
                score = get_score(current_set)
                if score < best_score:
                    best_score = score
                    best_set = list(current_set)
                    # If we hit 3 or 4, we can stop early? 3 is theoretical min for 12 distinct?
                    # Min Sidon size for 12 elements:
                    # Sidon set size k in [0, N] approx sqrt(N).
                    # In [0, 128], max Sidon is around sqrt(128) ~ 11? No, Sidon sets can be larger.
                    # But we are looking for the LARGEST Sidon SUBSET of our 12.
                    # If our set is "bad", the largest subset might be small.
                    # If best_score <= 4, it's very good.
                    if best_score <= 4:
                        break
        
        T *= cooling
        
    # Final polish: Local search on the best set found
    if time.time() < deadline - 5:
        current = sorted(best_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                if time.time() > deadline - 2:
                    break
                old_val = current[i]
                for new_val in range(129):
                    if new_val in current and new_val != old_val:
                        continue
                    if new_val == old_val:
                        continue
                    candidate = list(current)
                    candidate[i] = new_val
                    candidate = sorted(candidate)
                    score = max_sidon_subset_size(candidate)
                    if score < best_score:
                        best_score = score
                        best_set = candidate
                        improved = True
                        break
                if improved:
                    break
                    
    return sorted(best_set)
