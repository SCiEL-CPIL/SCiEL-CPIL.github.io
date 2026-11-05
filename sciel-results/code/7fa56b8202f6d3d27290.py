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

    def build_structured_initial():
        """Divide-and-conquer: partition into subcubes and solve each locally"""
        rng = np.random.default_rng(12345)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Divide into 9 subcubes based on first 2 coordinates
        for a in range(3):
            for b in range(3):
                subcube_indices = []
                for tail in itertools.product(range(3), repeat=6):
                    idx = a * powers[0] + b * powers[1]
                    for i, digit in enumerate(tail):
                        idx += digit * powers[2 + i]
                    subcube_indices.append(idx)
                
                remaining = [p for p in subcube_indices if not is_covered[p]]
                if not remaining:
                    continue
                
                while remaining:
                    best_idx = -1
                    best_count = -1
                    sample = remaining if len(remaining) <= 30 else rng.choice(remaining, 30, replace=False)
                    for p in sample:
                        count = np.sum(~is_covered[neighbors[p]])
                        if count > best_count:
                            best_count = count
                            best_idx = p
                    
                    if best_idx == -1:
                        break
                    
                    selected.append(best_idx)
                    for p in neighbors[best_idx]:
                        is_covered[p] = True
                    
                    remaining = [p for p in remaining if not is_covered[p]]
        
        if not np.all(is_covered):
            for b in range(total_points):
                if np.all(is_covered):
                    break
                if np.any(~is_covered[neighbors[b]]):
                    selected.append(b)
                    is_covered[neighbors[b]] = True
        
        return selected

    def fast_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        order = rng.permutation(total_points)
        
        for b in order:
            if np.all(is_covered):
                break
            uncovered_in_ball = np.sum(~is_covered[neighbors[b]])
            if uncovered_in_ball >= 5:
                selected.append(b)
                is_covered[neighbors[b]] = True
        
        if not np.all(is_covered):
            remaining_order = [b for b in order if b not in set(selected)]
            for b in remaining_order:
                if np.all(is_covered):
                    break
                is_covered[neighbors[b]] = True
                selected.append(b)
        
        return selected

    def prune(selected_indices, max_iter=100000):
        if not selected_indices:
            return []
            
        selected_list = list(selected_indices)
        if len(selected_list) == 0:
            return []
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        
        changed = True
        iter_count = 0
        while changed and iter_count < max_iter:
            changed = False
            iter_count += 1
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = neighbors[idx]
                if np.all(coverage_count[ball] > 1):
                    rem_idx = selected_list.pop(i)
                    coverage_count[neighbors[rem_idx]] -= 1
                    changed = True
                    break
        
        return selected_list

    def local_search_optimize(selected_indices, deadline):
        """Improved local search with multiple strategies"""
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        selected_set = set(selected_list)
        
        no_improve_count = 0
        max_no_improve = 4000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improvement_made = False
            
            for _ in range(100):
                if not selected_list:
                    break
                if time.time() > deadline:
                    break
                    
                c = selected_list[rng.integers(len(selected_list))]
                
                risky_points = neighbors[c][coverage_count[neighbors[c]] == 1]
                
                if len(risky_points) == 0:
                    coverage_count[neighbors[c]] -= 1
                    idx_in_list = selected_list.index(c)
                    selected_list.pop(idx_in_list)
                    selected_set.remove(c)
                    improvement_made = True
                    no_improve_count = 0
                    break
                else:
                    for pos in range(1, ball_size):
                        c_prime = neighbors[c][pos]
                        if c_prime in selected_set:
                            continue
                        
                        if np.all(np.isin(risky_points, neighbors[c_prime])):
                            coverage_count[neighbors[c]] -= 1
                            coverage_count[neighbors[c_prime]] += 1
                            
                            idx_in_list = selected_list.index(c)
                            selected_list[idx_in_list] = c_prime
                            selected_set.remove(c)
                            selected_set.add(c_prime)
                            
                            improvement_made = True
                            no_improve_count = 0
                            break
                    if improvement_made:
                        break
            
            if not improvement_made:
                no_improve_count += 1
            
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]
            
            if no_improve_count % 100 == 0 and no_improve_count > 0:
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
    deadline = start_time + 110.0
    
    best_code = None
    best_len = float('inf')
    
    if time.time() < deadline - 60:
        selected = build_structured_initial()
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
    
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 50:
            break
        selected = fast_greedy(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
    
    if best_code is None:
        best_code = build_structured_initial()
        best_code = prune(best_code)
        best_len = len(best_code)
    
    best_code = local_search_optimize(best_code, deadline)
    best_code = prune(best_code)
    
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
