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

    def luby_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            
            # Luby weight^2: pick with probability proportional to count^2
            weights = uncovered_counts.astype(np.float64)
            weights[weights == 0] = 0
            weights_sq = weights * weights
            total_w = weights_sq.sum()
            if total_w == 0:
                break
            probs = weights_sq / total_w
            best_idx = rng.choice(total_points, p=probs)
            
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
            # Vectorized: find all redundant codewords at once
            # A codeword idx is redundant if all points in its ball have coverage > 1
            # Check for each selected idx
            redundant_mask = np.zeros(len(selected_set), dtype=bool)
            sel_list = list(selected_set)
            for i, idx in enumerate(sel_list):
                if np.all(coverage_count[neighbors[idx]] > 1):
                    redundant_mask[i] = True
                    changed = True
            
            if changed:
                for i in range(len(sel_list)):
                    if redundant_mask[i]:
                        rem_idx = sel_list[i]
                        coverage_count[neighbors[rem_idx]] -= 1
                        selected_set.discard(rem_idx)
        
        return list(selected_set)

    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            # Try multiple random swaps per iteration
            for _ in range(5):
                if time.time() > deadline:
                    break
                    
                c = selected_list[rng.integers(len(selected_list))]
                c_prime = neighbors[c][rng.integers(1, ball_size)]
                
                if c_prime in selected_set:
                    continue
                
                # Tentatively swap c -> c_prime
                coverage_count[neighbors[c]] -= 1
                coverage_count[neighbors[c_prime]] += 1
                
                if np.any(coverage_count[neighbors[c]] == 0):
                    # Revert
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
                    continue
                else:
                    # Accept swap
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
        
        return best_code

    start_time = time.time()
    deadline = start_time + 115.0
    
    # Run greedy instances
    best_code = None
    best_len = float('inf')
    
    # Run more greedy instances since they're fast
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        selected = luby_greedy(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = luby_greedy(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search
    best_code = local_search_optimize(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
