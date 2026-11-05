import numpy as np
import time
import itertools

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

    # Pre-compute neighbors
    # neighbors[idx] contains idx and all 16 neighbors
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
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
            neighbors[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            neighbors[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def greedy_construct(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            # Calculate gains for all candidates
            # ~is_covered[neighbors] is a boolean matrix of shape (total_points, ball_size)
            # Summing along axis 1 gives the number of uncovered points covered by each candidate
            gains = np.sum(~is_covered[neighbors], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly
            c = candidates[rng.integers(len(candidates))]
            
            selected.append(c)
            is_covered[neighbors[c]] = True
            
        return selected

    def prune(selected_indices, neighbors, coverage_count=None):
        if not selected_indices:
            return [], np.zeros(total_points, dtype=np.int32)
            
        selected_list = list(selected_indices)
        if coverage_count is None:
            coverage_count = np.zeros(total_points, dtype=np.int32)
            for idx in selected_list:
                coverage_count[neighbors[idx]] += 1
        else:
            # Ensure coverage_count is consistent with selected_list
            # This is expensive, so we assume it's correct if passed from local search
            pass
            
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = neighbors[idx]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[ball] > 1):
                    coverage_count[ball] -= 1
                    selected_list.pop(i)
                    changed = True
                    break # Restart scan to handle cascading removals
                    
        return selected_list, coverage_count

    def local_search_optimize(selected_indices, deadline):
        selected_list, coverage_count = prune(selected_indices, neighbors)
        
        if not selected_list:
            return []
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        best_cov = coverage_count.copy()
        
        rng = np.random.default_rng(42)
        
        start_time = time.time()
        
        while time.time() < deadline:
            if not selected_list:
                break
                
            # Pick a random codeword to modify
            i = rng.integers(len(selected_list))
            c = selected_list[i]
            
            # Pick a random candidate c_prime
            # We can pick from the whole space or just neighbors.
            # Whole space allows larger jumps.
            c_prime = rng.integers(total_points)
            
            if c_prime == c:
                continue
                
            # Check if swap is valid
            # Temporarily remove c and add c_prime
            # We need to check if any point becomes uncovered
            # Points in neighbors[c] lose 1 count
            # Points in neighbors[c_prime] gain 1 count
            
            # Efficient check:
            # 1. Decrease counts for neighbors[c]
            # 2. Increase counts for neighbors[c_prime]
            # 3. Check if any count < 0
            # 4. Revert if invalid
            
            # To avoid copying the whole array, we can do this:
            # But modifying and reverting is O(17) each.
            
            # Let's just do it.
            neighbors_c = neighbors[c]
            neighbors_cp = neighbors[c_prime]
            
            # Save original values? No, just decrement, check, increment.
            # But we must be careful if there is overlap.
            
            # Decrement c
            coverage_count[neighbors_c] -= 1
            
            # Increment c_prime
            coverage_count[neighbors_cp] += 1
            
            # Check validity
            if np.any(coverage_count < 0):
                # Invalid, revert
                coverage_count[neighbors_cp] -= 1
                coverage_count[neighbors_c] += 1
                continue
            
            # Valid swap.
            # Perform swap in list
            selected_list[i] = c_prime
            
            # Try to prune
            pruned_list, pruned_cov = prune(selected_list, neighbors, coverage_count)
            
            if len(pruned_list) < best_len:
                # Improvement
                best_len = len(pruned_list)
                best_code = pruned_list[:]
                best_cov = pruned_cov.copy()
                
                # Update current state to the improved one
                selected_list = pruned_list
                coverage_count = pruned_cov
            else:
                # No improvement. 
                # Accept with some probability to explore?
                # If we revert, we stay in the same state.
                # If we accept, we move to a neutral state.
                # Neutral moves can help escape local minima.
                # Let's accept with probability 0.1
                if rng.random() < 0.1:
                    # Keep the swap (neutral)
                    # selected_list and coverage_count are already updated
                    pass
                else:
                    # Revert
                    selected_list[i] = c
                    # Revert coverage count
                    coverage_count[neighbors_cp] -= 1
                    coverage_count[neighbors_c] += 1

        return best_code

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_code = None
    best_len = float('inf')
    
    # Run greedy instances
    greedy_deadline = start_time + 20.0
    
    for seed in range(20):
        if time.time() > greedy_deadline:
            break
        selected = greedy_construct(seed)
        selected, _ = prune(selected, neighbors)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = greedy_construct(0)
        best_code, _ = prune(best_code, neighbors)
        best_len = len(best_code)
    
    # Local Search
    best_code = local_search_optimize(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
