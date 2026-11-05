import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each index
    neighbors = np.empty((total_words, ball_size), dtype=np.int32)
    for idx in range(total_words):
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

    def is_valid_code(code_indices):
        if not code_indices:
            return total_points == 0
        is_covered = np.zeros(total_words, dtype=bool)
        for idx in code_indices:
            is_covered[neighbors[idx]] = True
        return np.all(is_covered)

    def greedy_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_words, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            uncovered = ~is_covered
            counts = np.sum(uncovered[neighbors], axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.integers(len(candidates))]
            
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []
            
        selected_set = set(selected_indices)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        
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

    def local_search(selected_indices, deadline, seed=42):
        selected_list = prune(selected_indices)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(seed)
        
        # Build reverse mapping: for each point, which codewords cover it
        # This is expensive to maintain, so we'll use a different approach:
        # For each codeword, track which points it uniquely covers
        
        stuck_counter = 0
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            # Pick a random codeword to try removing
            c_idx = rng.integers(len(selected_list))
            c = selected_list[c_idx]
            
            # Tentatively remove c
            coverage_count[neighbors[c]] -= 1
            
            # Find points that are now uncovered
            uncovered_after_removal = np.where(coverage_count == 0)[0]
            
            if len(uncovered_after_removal) == 0:
                # c was redundant! Remove it
                selected_list.pop(c_idx)
                selected_set.discard(c)
                stuck_counter = 0
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                continue
            
            # We need to find a codeword that covers all uncovered points
            # or at least most of them, and doesn't introduce new problems
            
            # For each uncovered point, find which codewords cover it
            # This is expensive, so let's use a heuristic:
            # Pick a random uncovered point and find candidates that cover it
            
            if len(uncovered_after_removal) > 10:
                # Too many uncovered, revert
                coverage_count[neighbors[c]] += 1
                stuck_counter += 1
                continue
            
            # Try to find a single codeword that covers all uncovered points
            # For small sets, we can check intersection
            if len(uncovered_after_removal) == 1:
                # Only one point uncovered, find any codeword covering it
                target = uncovered_after_removal[0]
                # Find codewords whose ball includes target
                # We need reverse lookup: for each point, which codeword indices cover it
                # This is expensive to compute, so let's sample
                found = False
                for _ in range(20):
                    candidate = rng.integers(total_words)
                    if candidate in selected_set:
                        continue
                    # Check if candidate covers target
                    if target in neighbors[candidate]:
                        # Check if adding candidate doesn't break anything
                        # It won't break anything since we're adding, not removing
                        # But we need to ensure the swap is valid
                        # Actually, we removed c, so we need to add a replacement
                        coverage_count[neighbors[candidate]] += 1
                        # Check no point has count 0
                        if not np.any(coverage_count == 0):
                            # Accept: replace c with candidate
                            selected_list[c_idx] = candidate
                            selected_set.discard(c)
                            selected_set.add(candidate)
                            found = True
                            break
                        else:
                            coverage_count[neighbors[candidate]] -= 1
                
                if not found:
                    # Revert
                    coverage_count[neighbors[c]] += 1
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                continue
            
            # For multiple uncovered points, try to find a codeword that covers all
            # This is harder, so let's just revert for now
            coverage_count[neighbors[c]] += 1
            stuck_counter += 1
            
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_code = None
    best_len = float('inf')
    
    # Run greedy with multiple seeds
    for seed in range(30):
        if time.time() > deadline - 40:
            break
        selected = greedy_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = greedy_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search
    best_code = local_search(best_code, deadline, seed=42)
    
    # Random restarts for local search to escape local minima
    if time.time() < deadline - 10:
        for restart in range(3):
            if time.time() >= deadline - 5:
                break
            
            # Start from a copy of the best code
            rc_code = best_code[:]
            rc_size = len(rc_code)
            
            # Run local search with a different seed
            rc_code = local_search(rc_code, deadline - 5, seed=42 + restart * 1000)
            
            if len(rc_code) < best_len:
                best_len = len(rc_code)
                best_code = rc_code
    
    # Final Validation
    if not is_valid_code(best_code):
        # Fallback to the last known valid code from greedy
        print("Warning: Local search result invalid, falling back to greedy.")
        fallback_code = greedy_cover(123)
        fallback_code = prune(fallback_code)
        if is_valid_code(fallback_code):
            best_code = fallback_code
        else:
            # Absolute fallback: full space (valid but large)
            best_code = list(range(total_words))
            
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
