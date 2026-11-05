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

    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def is_valid_code(code_indices):
        if not code_indices:
            return total_points == 0
        is_covered = np.zeros(total_points, dtype=bool)
        for idx in code_indices:
            is_covered[neighbors[idx]] = True
        return np.all(is_covered)

    def luby_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            # Vectorized calculation of how many uncovered points each ball covers
            uncovered = ~is_covered
            counts = np.sum(uncovered[neighbors], axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Random tie-breaking among max
            candidates = np.where(counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = candidates[rng.integers(len(candidates))]
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []
            
        selected_set = set(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_set:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            sel_list = list(selected_set)
            to_remove = []
            for idx in sel_list:
                if np.all(coverage_count[neighbors[idx]] > 1):
                    to_remove.append(idx)
            if to_remove:
                changed = True
                for rem_idx in to_remove:
                    coverage_count[neighbors[rem_idx]] -= 1
                    selected_set.discard(rem_idx)
        
        return list(selected_set)

    def local_search_optimize(selected_indices, deadline):
        selected_list = prune(selected_indices)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 2.0:
            if not selected_list:
                break
            
            if no_improve_count > max_no_improve:
                # Restart from best
                if best_len < len(selected_list):
                    selected_list = best_code[:]
                    selected_set = set(selected_list)
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_list:
                        coverage_count[neighbors[idx]] += 1
                    no_improve_count = 0
                else:
                    break
            
            made_progress = False
            
            # Try random swaps
            c = selected_list[rng.integers(len(selected_list))]
            
            # Try all 16 neighbors of c (excluding itself)
            for pos in range(1, ball_size):
                c_prime = neighbors[c][pos]
                
                if c_prime in selected_set:
                    continue
                
                # Check if swap is valid
                # Points in neighbors[c] with count 1 are "risky"
                risky_points = neighbors[c][coverage_count[neighbors[c]] == 1]
                
                if len(risky_points) == 0:
                    # Safe swap
                    coverage_count[neighbors[c]] -= 1
                    coverage_count[neighbors[c_prime]] += 1
                    
                    pos_c = selected_list.index(c)
                    selected_list[pos_c] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    
                    made_progress = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    break
                else:
                    # Check if c_prime covers all risky points
                    if np.all(np.isin(risky_points, neighbors[c_prime])):
                        coverage_count[neighbors[c]] -= 1
                        coverage_count[neighbors[c_prime]] += 1
                        
                        pos_c = selected_list.index(c)
                        selected_list[pos_c] = c_prime
                        selected_set.discard(c)
                        selected_set.add(c_prime)
                        
                        made_progress = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        break
                    else:
                        no_improve_count += 1
            
            # Try removals if no swap worked
            if not made_progress:
                c = selected_list[rng.integers(len(selected_list))]
                if np.all(coverage_count[neighbors[c]] > 1):
                    coverage_count[neighbors[c]] -= 1
                    pos_c = selected_list.index(c)
                    selected_list.pop(pos_c)
                    selected_set.discard(c)
                    
                    made_progress = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            
            if not made_progress:
                no_improve_count += 1
                
        return best_code

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_code = None
    best_len = float('inf')
    
    # Run greedy instances
    # Vectorized greedy is fast, so we can run more
    greedy_deadline = deadline - 40.0  # Leave 40s for local search
    max_greedy = 20
    
    for seed in range(max_greedy):
        if time.time() > greedy_deadline:
            break
        selected = luby_greedy(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
        if best_len <= 410:
            break
    
    if best_code is None:
        best_code = luby_greedy(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search with remaining time
    remaining = deadline - time.time()
    if remaining > 5:
        best_code = local_search_optimize(best_code, deadline)
        
    # Final validation and fallback
    if not is_valid_code(best_code):
        # Fallback to tetracode product (486 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                # Convert word to index
                idx = 0
                for i, digit in enumerate(word):
                    idx += digit * powers[i]
                best_code_indices.append(idx)
        best_code = best_code_indices

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
