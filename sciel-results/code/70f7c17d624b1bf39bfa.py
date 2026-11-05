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
        
        # Precompute uncovered counts? No, dynamic.
        # We'll just pick the one that covers the most uncovered points.
        # To speed up, we can use a heap, but let's stick to vectorized sum for simplicity and robustness.
        # If it's too slow, we reduce iterations.
        
        while not np.all(is_covered):
            # Calculate how many uncovered points each candidate covers
            # ~is_covered is 1 where uncovered
            # neighbors[idx] gives the ball
            # We want sum of ~is_covered[neighbors[idx]]
            # This is a matrix-vector product: neighbors (one-hot?) no.
            # It's just indexing.
            
            uncovered = ~is_covered
            # Sum over the second axis for each row of neighbors
            # neighbors is (N, B). uncovered is (N,).
            # We need for each i, sum(uncovered[neighbors[i, :]])
            # This is equivalent to (neighbors_one_hot @ uncovered)
            # But we don't have one-hot.
            # We can use np.take or advanced indexing.
            # np.sum(uncovered[neighbors], axis=1)
            
            counts = np.sum(uncovered[neighbors], axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Pick random among those with max count
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
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            # Try random swaps
            # To make it faster, we can limit the number of swaps per second
            for _ in range(50):
                if time.time() > deadline:
                    break
                    
                c = selected_list[rng.integers(len(selected_list))]
                # Pick a random neighbor of c
                c_prime = neighbors[c][rng.integers(1, ball_size)]
                
                if c_prime in selected_set:
                    continue
                
                # Tentative swap
                coverage_count[neighbors[c]] -= 1
                coverage_count[neighbors[c_prime]] += 1
                
                # Check validity: no point in neighbors[c] should have count 0
                if np.any(coverage_count[neighbors[c]] == 0):
                    # Revert
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
                    continue
                
                # Accept swap
                # Update list
                pos = selected_list.index(c)
                selected_list[pos] = c_prime
                selected_set.discard(c)
                selected_set.add(c_prime)
                
                # Try to prune
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    selected_list = pruned
                    selected_set = set(selected_list)
                    # Rebuild coverage
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_list:
                        coverage_count[neighbors[idx]] += 1
                    
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                else:
                    # Keep swap if it doesn't increase size (it doesn't, it's a swap)
                    # But we might want to revert if it doesn't help?
                    # For now, keep it to explore.
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_code = None
    best_len = float('inf')
    
    # Run greedy
    for seed in range(20):
        if time.time() > deadline - 30:
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
    best_code = local_search(best_code, deadline)
    
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
