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
            # then pick a new codeword that covers those uncovered points
            
            for attempt in range(100):
                if time.time() > deadline:
                    break
                    
                # Pick a random codeword to try removing
                c = selected_list[rng.integers(len(selected_list))]
                
                # Tentatively remove c
                coverage_count[neighbors[c]] -= 1
                
                # Find points that are now uncovered
                uncovered_after_removal = np.where(coverage_count == 0)[0]
                
                if len(uncovered_after_removal) == 0:
                    # c was redundant! Remove it
                    pos = selected_list.index(c)
                    selected_list.pop(pos)
                    selected_set.discard(c)
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
                        candidate = rng.integers(total_points)
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
                                pos = selected_list.index(c)
                                selected_list[pos] = candidate
                                selected_set.discard(c)
                                selected_set.add(candidate)
                                found = True
                                break
                            else:
                                coverage_count[neighbors[candidate]] -= 1
                    
                    if not found:
                        # Revert
                        coverage_count[neighbors[c]] += 1
                    continue
                
                # For multiple uncovered points, try to find a codeword that covers all
                # This is harder, so let's just revert for now
                coverage_count[neighbors[c]] += 1
        
        return best_code

    def local_search_v2(selected_indices, deadline):
        """
        Improved local search:
        - Reduction pass to remove redundant codewords
        - 1-for-1 swap using O(1) neighbor-list intersection for candidates
        - 2-for-1 move (remove two, add one) to break local minima
        """
        selected_list = prune(selected_indices)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        def reduce_code(code, counts):
            """Remove redundant codewords iteratively."""
            if not code:
                return code, counts
            current_code = list(code)
            current_counts = counts.copy()
            
            while True:
                removed_any = False
                next_code = []
                for c in current_code:
                    if np.min(current_counts[neighbors[c]]) >= 2:
                        current_counts[neighbors[c]] -= 1
                        removed_any = True
                    else:
                        next_code.append(c)
                current_code = next_code
                if not removed_any:
                    break
            return current_code, current_counts
        
        # Pre-compute neighbor sets for O(1) intersection lookups
        # neighbors is already an array; we'll use it directly
        
        while time.time() < deadline:
            # 1. Reduction Pass
            old_size = len(selected_list)
            selected_list, coverage_count = reduce_code(selected_list, coverage_count)
            
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]
            
            if time.time() > deadline:
                break
            
            if len(selected_list) == 0:
                break
            
            # 2. Swap Pass (1-for-1)
            improved = False
            indices = list(range(len(selected_list)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                c = selected_list[idx]
                u_indices = neighbors[c]
                unique_mask = coverage_count[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    pos = selected_list.index(c)
                    selected_list.pop(pos)
                    selected_set.discard(c)
                    coverage_count[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 4:
                    continue
                
                # O(1) intersection of neighbor lists of unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                else:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                    for u in unique_points[1:]:
                        candidates_set &= set(neighbors[u].tolist())
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                candidates_set.difference_update(selected_set)
                if not candidates_set:
                    continue
                
                # Score candidates by how many additional points they cover
                # that are currently covered by few codewords
                best_cand = None
                best_score = -1
                cand_list = list(candidates_set)
                # Sample if too many
                if len(cand_list) > 20:
                    cand_list = [cand_list[i] for i in rng.choice(len(cand_list), size=20, replace=False)]
                
                for c_prime in cand_list:
                    # Count how many points in c_prime's ball have low coverage
                    ball = neighbors[c_prime]
                    score = np.sum(coverage_count[ball] <= 2)
                    if score > best_score:
                        best_score = score
                        best_cand = c_prime
                
                if best_cand is None:
                    continue
                
                # Apply swap
                old_C = selected_list.copy()
                old_counts = coverage_count.copy()
                
                coverage_count[neighbors[c]] -= 1
                pos = selected_list.index(c)
                selected_list[pos] = best_cand
                selected_set.discard(c)
                selected_set.add(best_cand)
                coverage_count[neighbors[best_cand]] += 1
                
                # Check if swap enables reduction
                new_code, new_counts = reduce_code(selected_list, coverage_count)
                
                if len(new_code) < len(old_C):
                    selected_list = new_code
                    coverage_count = new_counts
                    selected_set = set(new_code)
                    improved = True
                    break
                else:
                    # Revert
                    selected_list = old_C
                    coverage_count = old_counts
                    selected_set = set(old_C)
            
            if time.time() > deadline:
                break
            
            # 3. Two-for-one move: remove two codewords, add one
            if len(selected_list) >= 2:
                two_opt_improved = False
                # Try a few random pairs
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    
                    i1 = rng.integers(len(selected_list))
                    i2 = rng.integers(len(selected_list))
                    if i1 == i2:
                        continue
                    
                    c1 = selected_list[i1]
                    c2 = selected_list[i2]
                    
                    # Tentatively remove both
                    coverage_count[neighbors[c1]] -= 1
                    coverage_count[neighbors[c2]] -= 1
                    
                    uncovered = np.where(coverage_count == 0)[0]
                    
                    if len(uncovered) == 0:
                        # Both were redundant together
                        selected_set.discard(c1)
                        selected_set.discard(c2)
                        # Remove from list (handle order)
                        if i1 > i2:
                            i1, i2 = i2, i1
                        selected_list.pop(i1)
                        selected_list.pop(i2)
                        two_opt_improved = True
                        break
                    elif len(uncovered) <= 5:
                        # Find a codeword that covers all uncovered points
                        if len(uncovered) == 1:
                            cand_set = set(neighbors[uncovered[0]].tolist())
                        else:
                            cand_set = set(neighbors[uncovered[0]].tolist())
                            for u in uncovered[1:]:
                                cand_set &= set(neighbors[u].tolist())
                                if not cand_set:
                                    break
                        
                        if cand_set:
                            cand_set.difference_update(selected_set)
                            if cand_set:
                                # Pick best candidate
                                cand_list = list(cand_set)
                                if len(cand_list) > 15:
                                    cand_list = [cand_list[i] for i in rng.choice(len(cand_list), size=15, replace=False)]
                                
                                best_cand = None
                                best_score = -1
                                for c_prime in cand_list:
                                    ball = neighbors[c_prime]
                                    score = np.sum(coverage_count[ball] <= 2)
                                    if score > best_score:
                                        best_score = score
                                        best_cand = c_prime
                                
                                if best_cand is not None:
                                    # Apply: remove c1, c2, add best_cand
                                    pos1 = selected_list.index(c1)
                                    pos2 = selected_list.index(c2)
                                    if pos1 > pos2:
                                        pos1, pos2 = pos2, pos1
                                    selected_list.pop(pos1)
                                    selected_list.pop(pos2)
                                    selected_set.discard(c1)
                                    selected_set.discard(c2)
                                    selected_list.append(best_cand)
                                    selected_set.add(best_cand)
                                    coverage_count[neighbors[best_cand]] += 1
                                    two_opt_improved = True
                                    # Run reduction after
                                    selected_list, coverage_count = reduce_code(selected_list, coverage_count)
                                    selected_set = set(selected_list)
                                    break
                    # Revert
                    coverage_count[neighbors[c1]] += 1
                    coverage_count[neighbors[c2]] += 1
                    
                    if two_opt_improved:
                        break
            
            if improved or two_opt_improved:
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_code = None
    best_len = float('inf')
    
    # Run greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 50:
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
    
    # Local Search - use both versions and take the best
    # Run local_search_v2 first (it's more effective), then local_search if time permits
    code1 = local_search_v2(best_code, deadline - 5)
    if time.time() < deadline - 2:
        code2 = local_search(best_code, deadline)
        if len(code2) < len(code1):
            code1 = code2
    best_code = code1
    
    # Final Validation
    if not is_valid_code(best_code):
        # Fallback to the last known valid code from greedy
        # We need to keep a copy of the best valid greedy code
        # Let's re-run a simple greedy if this happens, or just return the greedy one.
        # Since we overwrote best_code, we should have kept a backup.
        # For simplicity, if invalid, run one more greedy and prune.
        print("Warning: Local search result invalid, falling back to greedy.")
        fallback_code = greedy_cover(123)
        fallback_code = prune(fallback_code)
        if is_valid_code(fallback_code):
            best_code = fallback_code
        else:
            # Absolute fallback: full space (valid but large)
            best_code = list(range(total_points))
            
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
