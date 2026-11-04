import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
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

    # Pre-compute neighbors (the ball of radius 1) for each point
    # neighbors[idx] is a list of 17 indices
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        
        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            
            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)
            
        neighbors[idx] = nbrs

    # Convert neighbors to numpy arrays for faster vectorized operations where possible
    # But lists are fine for iteration.
    # Let's create a set of all neighbor pairs for quick lookup? No, just use the lists.
    
    # Helper to check if a set of indices covers all points
    def covers_all(selected_indices):
        covered = np.zeros(total_points, dtype=bool)
        for idx in selected_indices:
            covered[neighbors[idx]] = True
        return np.all(covered)

    # Greedy Set Cover
    def greedy_set_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Precompute all balls as a 2D array for vectorized counting
        # balls_matrix[set_idx] = array of points covered by set_idx
        balls_matrix = np.array(neighbors, dtype=np.int32)
        
        while not np.all(is_covered):
            # Count uncovered points for each set
            # ~is_covered is a boolean vector of length total_points
            # balls_matrix is (total_points, ball_size)
            # We want sum(~is_covered[balls_matrix[i]] for each i)
            # This is equivalent to matrix multiplication of (~is_covered) as row vector and balls_matrix?
            # No, balls_matrix[i, j] is point index.
            # We need: for each row i, count how many elements in row i are in the set of uncovered points.
            # uncovered_mask = ~is_covered
            # counts[i] = sum(uncovered_mask[balls_matrix[i, :]])
            
            # This can be done efficiently with numpy
            # But doing it from scratch every iteration is O(N * B) = 6561 * 17 ~ 110k ops.
            # Number of iterations ~ 400-500. Total ~ 50M ops. Acceptable in numpy.
            
            uncovered_mask = ~is_covered
            # Efficient way: 
            # For each point p, if it is uncovered, it contributes 1 to the count of all sets covering p.
            # So we can maintain an array `gain` of length total_points.
            # Initially, gain[i] = ball_size (17) for all i.
            # When a point p becomes covered, we decrement gain[i] for all sets i that cover p.
            
            # Let's implement the incremental gain approach for speed.
            # Reset gain array
            gain = np.full(total_points, ball_size, dtype=np.int32)
            
            # We need to know, for each point p, which sets cover it.
            # This is the "dual" of balls_matrix.
            # coverers[p] = list of set indices i such that p is in neighbors[i].
            # Since neighbors[i] includes i itself and its 16 neighbors,
            # coverers[p] includes p and all neighbors of p.
            # So coverers[p] is exactly neighbors[p]!
            # Wait, if p is in neighbors[i], does i imply p? Yes, distance is symmetric.
            # So the sets covering point p are exactly the points in neighbors[p].
            
            # So, when a point p becomes covered, we decrement gain for all i in neighbors[p].
            
            # Let's re-implement the greedy loop with this optimization.
            
            selected = []
            is_covered = np.zeros(total_points, dtype=bool)
            gain = np.full(total_points, ball_size, dtype=np.int32)
            
            while not np.all(is_covered):
                # Find set with max gain
                max_gain = np.max(gain)
                # Random tie-breaking
                candidates = np.where(gain == max_gain)[0]
                if len(candidates) > 1:
                    best_idx = rng.choice(candidates)
                else:
                    best_idx = candidates[0]
                    
                selected.append(best_idx)
                
                # Update is_covered and gain
                # Points covered by best_idx are neighbors[best_idx]
                new_points = neighbors[best_idx]
                
                # For each point p in new_points that is NOT yet covered:
                #   1. Mark as covered
                #   2. Decrement gain for all sets covering p (which are neighbors[p])
                
                # Vectorized update:
                # We need to update gain for all neighbors of all newly covered points.
                # This is a bit tricky to vectorize fully without loops, but we can loop over the 17 points.
                
                for p in new_points:
                    if not is_covered[p]:
                        is_covered[p] = True
                        # Decrement gain for all sets covering p
                        gain[neighbors[p]] -= 1
                        
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return selected_indices
            
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[neighbors[idx]] > 1):
                    # Remove
                    rem_idx = selected_list.pop(i)
                    coverage_count[neighbors[rem_idx]] -= 1
                    changed = True
                    break # Restart scan after change
        return selected_list

    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        
        # Initial prune
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        while time.time() < deadline:
            # Try to prune
            pruned = prune(selected_list)
            if len(pruned) < len(selected_list):
                selected_list = pruned
                # Recalculate counts
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[neighbors[idx]] += 1
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                continue
            
            # Random Valid Move (Random Walk)
            # Pick random index i, random neighbor c_prime of selected_list[i]
            # If swap is valid, accept.
            
            n_selected = len(selected_list)
            if n_selected == 0:
                break
                
            i = rng.integers(0, n_selected)
            c = selected_list[i]
            
            # Pick a random neighbor
            c_prime = neighbors[c][rng.integers(0, len(neighbors[c]))]
            
            if c_prime == c:
                continue
                
            # Check if c_prime is already in selected_list
            # This check is O(N). We can optimize with a set, but N is small (~400).
            if c_prime in selected_list:
                continue
                
            # Tentatively swap
            # Decrement counts for c, Increment for c_prime
            coverage_count[neighbors[c]] -= 1
            coverage_count[neighbors[c_prime]] += 1
            
            # Check validity: no count should be 0
            # We only need to check points in neighbors[c] U neighbors[c_prime]
            # But checking the whole array is O(6561).
            # Checking only the union is faster.
            
            # Create a set of points to check? Or just check the two balls.
            # If a point in neighbors[c] \ neighbors[c_prime] becomes 0, invalid.
            # If a point in neighbors[c_prime] \ neighbors[c] becomes 0, invalid (but it was 0 before, so it can't become 0 by incrementing? Wait.
            # Before swap: count(p) >= 1 for all p.
            # After swap:
            #   For p in neighbors[c] \ neighbors[c_prime]: count(p) decreases by 1. Must be >= 1.
            #   For p in neighbors[c_prime] \ neighbors[c]: count(p) increases by 1. Always >= 1.
            #   For p in intersection: count(p) unchanged.
            # So we only need to check points in neighbors[c] \ neighbors[c_prime].
            
            valid = True
            # Efficient check:
            # We can just check if min(coverage_count[neighbors[c]]) >= 1?
            # No, points in neighbors[c_prime] \ neighbors[c] are not in neighbors[c].
            # But points in neighbors[c] \ neighbors[c_prime] are in neighbors[c].
            # So if min(coverage_count[neighbors[c]]) >= 1, then all points in neighbors[c] are covered.
            # This includes the intersection and the difference.
            # Is that sufficient?
            # Yes, because points NOT in neighbors[c] are either in neighbors[c_prime] (increased) or neither (unchanged).
            # So checking neighbors[c] is sufficient.
            
            if np.min(coverage_count[neighbors[c]]) < 1:
                # Invalid, revert
                coverage_count[neighbors[c_prime]] -= 1
                coverage_count[neighbors[c]] += 1
            else:
                # Valid, accept
                selected_list[i] = c_prime
                # Try to prune after swap
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    selected_list = pruned
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_list:
                        coverage_count[neighbors[idx]] += 1
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]

        return best_code

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    start_time = time.time()
    deadline = start_time + 110.0
    
    # Run a few greedy instances and pick the best after pruning
    best_code = None
    best_len = float('inf')
    
    # Run 5 greedy instances
    for seed in range(5):
        if time.time() > deadline - 30: # Leave 30s for local search
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = greedy_set_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search on the best code
    best_code = local_search_optimize(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
