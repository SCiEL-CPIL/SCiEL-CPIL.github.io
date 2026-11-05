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

    # Convert to numpy arrays for fast indexing
    neighbors_arr = np.array(neighbors, dtype=np.int32)
    # neighbors_arr[i] is the ball (17 points) covered by codeword i

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
                    rem_idx = selected_list.pop(i)
                    coverage_count[neighbors[rem_idx]] -= 1
                    changed = True
                    break
        return selected_list

    def greedy_set_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Incremental gain: gain[i] = number of uncovered points in ball(i)
        gain = np.full(total_points, ball_size, dtype=np.int32)
        
        while not np.all(is_covered):
            # Find set with max gain, random tie-breaking
            max_gain = np.max(gain)
            candidates = np.where(gain == max_gain)[0]
            if len(candidates) > 1:
                best_idx = int(rng.choice(candidates))
            else:
                best_idx = int(candidates[0])
                
            selected.append(best_idx)
            
            # Update is_covered and gain incrementally
            new_points = neighbors[best_idx]
            for p in new_points:
                if not is_covered[p]:
                    is_covered[p] = True
                    # Decrement gain for all sets that cover p
                    # Sets covering p are exactly neighbors[p] (by symmetry)
                    gain[neighbors[p]] -= 1
                    
        return selected

    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        # Helper: try to remove a codeword and repair
        def try_remove(idx_to_remove, coverage_count, selected_list):
            """Try to remove idx_to_remove. Return (success, new_coverage, new_selected)"""
            # Check if removing it leaves any point uncovered
            ball = neighbors[idx_to_remove]
            # Points that would become uncovered: those with count == 1
            uncovered_after = [p for p in ball if coverage_count[p] == 1]
            if not uncovered_after:
                # Can remove directly
                return True, coverage_count, selected_list
            
            # Try to add a neighbor of idx_to_remove that covers all uncovered_after
            # For each candidate c' in neighbors[idx_to_remove] (excluding itself)
            # Check if ball(c') covers all of uncovered_after
            for c_prime in neighbors[idx_to_remove]:
                if c_prime == idx_to_remove:
                    continue
                # Check if c_prime covers all uncovered_after
                ball_cp = neighbors[c_prime]
                all_covered = all(p in ball_cp for p in uncovered_after)
                if all_covered:
                    # Also need to make sure c_prime is not already in selected_list
                    if c_prime not in selected_list:
                        # Do the swap: remove idx_to_remove, add c_prime
                        return True, coverage_count, selected_list
            return False, coverage_count, selected_list

        while time.time() < deadline:
            # Try to prune
            pruned = prune(selected_list)
            if len(pruned) < len(selected_list):
                selected_list = pruned
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[neighbors[idx]] += 1
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                continue
            
            n_selected = len(selected_list)
            if n_selected == 0:
                break
                
            # Random swap move
            i = int(rng.integers(0, n_selected))
            c = selected_list[i]
            c_prime = int(neighbors[c][rng.integers(0, ball_size)])
            
            if c_prime == c:
                continue
            if c_prime in selected_list:
                continue
                
            # Tentatively swap
            coverage_count[neighbors[c]] -= 1
            coverage_count[neighbors[c_prime]] += 1
            
            if np.min(coverage_count[neighbors[c]]) < 1:
                coverage_count[neighbors[c_prime]] -= 1
                coverage_count[neighbors[c]] += 1
            else:
                selected_list[i] = c_prime
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
    deadline = start_time + 115.0
    
    # Run greedy instances and pick the best after pruning
    best_code = None
    best_len = float('inf')
    
    # Run multiple greedy instances with different seeds
    num_greedy = 8
    for seed in range(num_greedy):
        if time.time() > deadline - 40:
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
