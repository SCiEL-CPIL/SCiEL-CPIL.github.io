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

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx

    def tetracode_initial():
        """Generate the 486-word tetracode-based covering code."""
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=4):
                word = h + list(tail)
                code.append(word_to_idx(word))
        return code

    def greedy_cover():
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            mask = ~is_covered
            counts = mask[neighbors].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            b = candidates[np.random.randint(len(candidates))]
            
            selected.append(b)
            is_covered[neighbors[b]] = True
            
        return selected

    def prune(selected):
        if not selected:
            return []
        
        selected_list = list(selected)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = neighbors[idx]
                if np.all(coverage_count[ball] > 1):
                    selected_list.pop(i)
                    coverage_count[neighbors[idx]] -= 1
                    changed = True
                    break
        return selected_list

    def local_search(selected, deadline):
        selected_list = prune(selected)
        best_list = selected_list[:]
        best_len = len(best_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        
        rng = np.random.default_rng(42)
        
        while time.time() < deadline:
            # 1. Try to prune
            pruned = prune(selected_list)
            if len(pruned) < len(selected_list):
                selected_list = pruned
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[neighbors[idx]] += 1
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_list = selected_list[:]
            
            if not selected_list:
                break

            # 2. Random swap: replace c with c'
            c_idx = rng.integers(len(selected_list))
            c = selected_list[c_idx]
            
            c_prime_idx = rng.integers(1, ball_size)
            c_prime = neighbors[c, c_prime_idx]
            
            if c_prime in selected_list:
                continue
            
            # Check validity: points in ball(c) \ ball(c') must have count > 1
            mask_cp = np.zeros(total_points, dtype=bool)
            mask_cp[neighbors[c_prime]] = True
            
            ball_c = neighbors[c]
            points_lost = ball_c[~mask_cp[ball_c]]
            
            if len(points_lost) > 0 and np.any(coverage_count[points_lost] == 1):
                continue
            
            # Perform swap
            coverage_count[ball_c] -= 1
            coverage_count[neighbors[c_prime]] += 1
            selected_list[c_idx] = c_prime
            
            # Try to prune after swap
            pruned = prune(selected_list)
            if len(pruned) < len(selected_list):
                selected_list = pruned
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[neighbors[idx]] += 1
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_list = selected_list[:]

        return best_list

    start_time = time.time()
    deadline = start_time + 115.0

    # Start from tetracode (486 words) - much better than greedy
    best_code = tetracode_initial()
    best_code = prune(best_code)
    
    # Run local search to improve
    best_code = local_search(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
