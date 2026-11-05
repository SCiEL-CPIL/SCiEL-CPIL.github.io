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

    def is_valid_code(code_indices):
        if not code_indices:
            return total_points == 0
        is_covered = np.zeros(total_points, dtype=bool)
        for idx in code_indices:
            is_covered[neighbors[idx]] = True
        return np.all(is_covered)

    def greedy_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
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

    def local_search(selected_indices, deadline):
        selected_list = prune(selected_indices)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        # Build reverse mapping: for each point, which codewords cover it
        # We'll maintain this dynamically
        # For each point, store list of codeword indices that cover it
        # This is expensive to maintain, so we'll use a different approach:
        # For each codeword, track which points it uniquely covers
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            # Find codewords that are candidates for removal (have unique coverage)
            # A codeword can be removed if all points in its ball are covered by others
            # i.e., coverage_count > 1 for all points in its ball
            
            # Try to find a codeword to remove and a replacement
            # Strategy: pick a codeword, remove it, find uncovered points,
            # then find a new codeword that covers those uncovered points
            
            # First, try to find a removable codeword
            removable_candidates = []
            for idx in selected_list:
                if np.all(coverage_count[neighbors[idx]] > 1):
                    removable_candidates.append(idx)
            
            if removable_candidates:
                # Pick one to remove
                rem_idx = removable_candidates[rng.integers(len(removable_candidates))]
                
                # Tentatively remove it
                coverage_count[neighbors[rem_idx]] -= 1
                selected_set.discard(rem_idx)
                pos = selected_list.index(rem_idx)
                selected_list[pos] = None  # Mark as removed
                
                # Find uncovered points
                uncovered = np.where(coverage_count == 0)[0]
                
                if len(uncovered) > 0:
                    # Find a new codeword that covers the most uncovered points
                    # For each uncovered point, find codewords that cover it
                    # and pick the one that covers the most uncovered points
                    
                    # For each uncovered point, get its neighbors (codewords that could cover it)
                    # Actually, we need to find a codeword whose ball covers uncovered points
                    # A codeword c covers point p if p is in neighbors[c]
                    # Equivalently, c is in neighbors[p]
                    
                    # For each uncovered point p, the codewords that cover it are neighbors[p]
                    # We want to find a codeword c that maximizes the number of uncovered points it covers
                    
                    # Build a frequency count: for each codeword c, how many uncovered points does it cover?
                    # This is equivalent to: for each uncovered point p, increment count[c] for c in neighbors[p]
                    
                    new_count = np.zeros(total_points, dtype=np.int32)
                    for p in uncovered:
                        new_count[neighbors[p]] += 1
                    
                    # Exclude already selected codewords
                    for idx in selected_list:
                        if idx is not None:
                            new_count[idx] = 0
                    
                    max_new = np.max(new_count)
                    if max_new > 0:
                        candidates = np.where(new_count == max_new)[0]
                        new_idx = candidates[rng.integers(len(candidates))]
                        
                        # Add the new codeword
                        coverage_count[neighbors[new_idx]] += 1
                        selected_list[pos] = new_idx
                        selected_set.add(new_idx)
                        
                        # Try to prune
                        pruned = prune(selected_list)
                        if len(pruned) < len(selected_list):
                            selected_list = pruned
                            selected_set = set(selected_list)
                            coverage_count = np.zeros(total_points, dtype=np.int32)
                            for idx in selected_list:
                                coverage_count[neighbors[idx]] += 1
                        
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                    else:
                        # Revert
                        coverage_count[neighbors[rem_idx]] += 1
                        selected_list[pos] = rem_idx
                        selected_set.add(rem_idx)
                else:
                    # No uncovered points, keep the removal
                    selected_list = [x for x in selected_list if x is not None]
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
            else:
                # No removable codewords, try a random swap to escape local optimum
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    
                    c = selected_list[rng.integers(len(selected_list))]
                    c_prime = neighbors[c][rng.integers(1, ball_size)]
                    
                    if c_prime in selected_set:
                        continue
                    
                    # Tentative swap
                    coverage_count[neighbors[c]] -= 1
                    coverage_count[neighbors[c_prime]] += 1
                    
                    if np.any(coverage_count[neighbors[c]] == 0):
                        coverage_count[neighbors[c_prime]] -= 1
                        coverage_count[neighbors[c]] += 1
                        continue
                    
                    pos = selected_list.index(c)
                    selected_list[pos] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    
                    # Try to prune
                    pruned = prune(selected_list)
                    if len(pruned) < len(selected_list):
                        selected_list = pruned
                        selected_set = set(selected_list)
                        coverage_count = np.zeros(total_points, dtype=np.int32)
                        for idx in selected_list:
                            coverage_count[neighbors[idx]] += 1
                        
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                    else:
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_code = None
    best_len = float('inf')
    
    # Run greedy
    for seed in range(15):
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
    
    # Local Search with multiple restarts
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        best_code = local_search(best_code, deadline)
        
        # Occasionally restart from a fresh greedy solution
        if restart < 2 and time.time() < deadline - 30:
            fresh = greedy_cover(100 + restart)
            fresh = prune(fresh)
            if len(fresh) < best_len:
                best_len = len(fresh)
                best_code = fresh
    
    # Final Validation
    if not is_valid_code(best_code):
        print("Warning: Local search result invalid, falling back to greedy.")
        fallback_code = greedy_cover(123)
        fallback_code = prune(fallback_code)
        if is_valid_code(fallback_code):
            best_code = fallback_code
        else:
            best_code = list(range(total_points))
            
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
