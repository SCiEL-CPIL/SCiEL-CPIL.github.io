import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def are_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

    def has_four_collinear(points):
        n = len(points)
        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(p1, p2, points[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False

    def calculate_score(points):
        """
        Exact branch and bound for the largest subset with no 3 collinear points.
        Uses bitmask representation for efficiency.
        """
        n = len(points)
        
        # Precompute for each pair (i, j) the set of k such that i, j, k are collinear.
        # pair_conflicts[i][j] will be a bitmask of such k's.
        pair_conflicts = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                pair_conflicts[i][j] = mask
                pair_conflicts[j][i] = mask

        best_score = 0
        full_mask = (1 << n) - 1

        def backtrack(available, current_size, selected_mask):
            nonlocal best_score
            
            # Pruning 1: Upper bound on size
            # If current_size + number of available points <= best_score, prune
            if current_size + bin(available).count('1') <= best_score:
                return
            
            if available == 0:
                if current_size > best_score:
                    best_score = current_size
                return

            # Heuristic: Pick a point from available.
            # To improve pruning, pick a point that is "constrained".
            # A simple heuristic: pick the first available point.
            i_bit = available & -available
            i = i_bit.bit_length() - 1
            
            # Option 1: Skip i
            backtrack(available ^ i_bit, current_size, selected_mask)
            
            # Option 2: Include i
            new_selected = selected_mask | i_bit
            new_available = available ^ i_bit
            
            # Prune new_available:
            # If we include i, then for any j in new_available, 
            # if there exists k in selected_mask such that i, j, k are collinear,
            # then j cannot be selected.
            
            forbidden = 0
            temp_sel = selected_mask
            while temp_sel:
                k_bit = temp_sel & -temp_sel
                k = k_bit.bit_length() - 1
                forbidden |= pair_conflicts[i][k]
                temp_sel ^= k_bit
            
            new_available &= ~forbidden
            
            backtrack(new_available, current_size + 1, new_selected)

        backtrack(full_mask, 0, 0)
        return best_score

    # Generate a pool of points with high collinearity density
    # Using a small grid ensures many collinear triples
    pool = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            pool.append([x, y])
    
    # Shuffle pool to randomize selection
    random.shuffle(pool)
    
    # We want to select 20 points from the pool.
    # Initialize with the first 20 points from the pool.
    # We need to ensure no 4 collinear in the initial set? 
    # The local search will fix it, but starting with a valid set is better.
    
    def find_initial_subset(pool):
        subset = []
        seen = set()
        for p in pool:
            if len(subset) >= 20:
                break
            # Check if adding p creates 4 collinear
            valid = True
            if p in seen:
                continue
            seen.add(p)
            test_subset = subset + [p]
            if has_four_collinear(test_subset):
                valid = False
            if valid:
                subset.append(p)
            else:
                seen.remove(p)
        
        # If we didn't get 20, fill with random points from pool
        if len(subset) < 20:
            seen_subset = set(tuple(p) for p in subset)
            for p in pool:
                if len(subset) >= 20:
                    break
                if tuple(p) not in seen_subset:
                    # Check 4 collinear
                    test_subset = subset + [p]
                    if not has_four_collinear(test_subset):
                        subset.append(p)
                        seen_subset.add(tuple(p))
        return subset

    best_subset = find_initial_subset(pool)
    if not best_subset:
        # Fallback: parabola
        best_subset = [[x, x*x] for x in range(-10, 10)]
    
    current_score = calculate_score(best_subset)
    best_score = current_score

    # Local Search: Swap points in best_subset with points in pool
    # We maintain a set of points in the current subset for fast lookup
    subset_set = set(tuple(p) for p in best_subset)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(20))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            old_point = best_subset[i]
            old_tuple = tuple(old_point)
            
            # Try swapping with points in the pool
            # To speed up, we can just iterate through the pool
            for new_point in pool:
                if time.time() > deadline:
                    break
                
                new_tuple = tuple(new_point)
                if new_tuple == old_tuple:
                    continue
                if new_tuple in subset_set:
                    continue
                
                # Construct new subset
                new_subset = best_subset[:]
                new_subset[i] = new_point
                
                # Check 4 collinear
                if has_four_collinear(new_subset):
                    continue
                
                # Calculate score
                score = calculate_score(new_subset)
                
                if score < best_score:
                    best_score = score
                    best_subset = new_subset
                    subset_set.remove(old_tuple)
                    subset_set.add(new_tuple)
                    improved = True
                    break # Restart local search with new best
                elif score == best_score and random.random() < 0.1:
                    # Accept equal score with small probability to escape local minima
                    best_subset = new_subset
                    subset_set.remove(old_tuple)
                    subset_set.add(new_tuple)
                    improved = True
                    break

    return best_subset

# Keep the entry point as required
# The task says "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.
