"""
Optimized Ternary Covering Code K_3(8,1) Solver.
Uses Randomized Greedy Set Cover with multiple restarts and aggressive pruning.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    if q != 3 or n != 8 or radius != 1:
        # Fallback to simple construction if parameters differ
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Pre-compute coverage sets
    # coverage_sets[i] contains the indices of the ball around word i
    coverage_sets = np.empty((total_points, ball_size), dtype=np.int32)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Pre-compute digits for all words to speed up neighbor generation
    # Or just compute on the fly, it's fast enough for 6561 * 16 ops
    for idx in range(total_points):
        # Decode word
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        coverage_sets[idx, 0] = idx
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
            coverage_sets[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            coverage_sets[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1

    def run_greedy(seed):
        """Runs a single randomized greedy set cover pass."""
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            # Calculate uncovered counts
            # ~is_covered[coverage_sets] is a boolean matrix (6561, 17)
            # Sum along axis 1 gives count of uncovered points for each candidate
            uncovered_counts = np.sum(~is_covered[coverage_sets], axis=1)
            
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
                
            # Find all candidates with max count
            candidates = np.where(uncovered_counts == max_count)[0]
            
            # Random choice among best candidates
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            is_covered[coverage_sets[best_idx]] = True
            
        return selected

    def prune(selected_indices):
        """Removes redundant codewords from the selected set."""
        if len(selected_indices) == 0:
            return selected_indices
            
        # Convert to list for easier manipulation
        selected_list = list(selected_indices)
        
        # Count coverage for each point
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[coverage_sets[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            # Find a redundant codeword
            # A codeword is redundant if no point in its ball has coverage_count == 1
            redundant_idx = -1
            for i, idx in enumerate(selected_list):
                ball = coverage_sets[idx]
                if np.all(coverage_count[ball] > 1):
                    redundant_idx = i
                    break
                    
            if redundant_idx != -1:
                # Remove it
                rem_idx = selected_list.pop(redundant_idx)
                coverage_count[coverage_sets[rem_idx]] -= 1
                changed = True
                
        return selected_list

    # Main Loop: Multiple Randomized Greedy Runs
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_code = None
    best_len = float('inf')
    iteration = 0
    
    # Run at least one standard greedy (seed 0)
    # Then run randomized versions
    while time.time() < deadline:
        # Use iteration as seed for diversity
        selected = run_greedy(iteration)
        
        # Prune the result
        selected = prune(selected)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
        # Optional: Early exit if we hit the theoretical lower bound (403) or known best (486)
        if best_len <= 486:
            break
            
        iteration += 1
        
        # If we have done many iterations and aren't improving, we might slow down?
        # No, each iteration is independent.
        
    # Final check: if best_code is None (shouldn't happen), fallback to first run
    if best_code is None:
        best_code = run_greedy(0)
        best_code = prune(best_code)

    # Convert indices to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result

# EVOLVE-BLOCK-END
