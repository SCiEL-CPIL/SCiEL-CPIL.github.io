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
    # neighbors[idx] is a list of 17 indices
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
        
        # Pre-compute reverse mapping: for each point, which balls contain it?
        # Actually, we can maintain uncovered_counts directly.
        # uncovered_counts[b] = number of uncovered points in ball b.
        # Initially, all balls have ball_size uncovered points.
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        
        # Pre-compute reverse neighbors: rev_neighbors[p] = list of ball indices containing p.
        # This is needed to update uncovered_counts efficiently when a point becomes covered.
        # But building this is O(total_points * ball_size) which is fine.
        # However, we can also just do the vectorized sum if it's fast enough.
        # Let's try the incremental approach first for speed.
        
        # Build reverse mapping
        rev_neighbors = [[] for _ in range(total_points)]
        for b in range(total_points):
            for p in neighbors[b]:
                rev_neighbors[p].append(b)
        
        while np.any(uncovered_counts > 0):
            # Find max uncovered count
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            
            # Random tie-breaking among max
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            
            # Update: for each point p in neighbors[best_idx] that was uncovered,
            # decrement uncovered_counts for all balls containing p.
            for p in neighbors[best_idx]:
                if not is_covered[p]:
                    is_covered[p] = True
                    for b_prime in rev_neighbors[p]:
                        uncovered_counts[b_prime] -= 1
            
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []
            
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            for i, idx in enumerate(selected_list):
                ball = neighbors[idx]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[ball] > 1):
                    rem_idx = selected_list.pop(i)
                    coverage_count[neighbors[rem_idx]] -= 1
                    changed = True
                    break # Restart scan
        return selected_list

    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        # Pre-compute a set for fast membership check
        selected_set = set(selected_list)
        
        # Simulated annealing parameters
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.999
        
        while time.time() < deadline:
            if not selected_list:
                break
                
            c = selected_list[rng.integers(len(selected_list))]
            
            # Try all neighbors of c as potential replacements
            for pos in range(1, ball_size):
                c_prime = neighbors[c][pos]
                
                if c_prime in selected_set:
                    continue
                
                # Check if removing c leaves any point uncovered
                risky_points = neighbors[c][coverage_count[neighbors[c]] == 1]
                
                if len(risky_points) == 0:
                    # Safe swap
                    coverage_count[neighbors[c]] -= 1
                    coverage_count[neighbors[c_prime]] += 1
                    
                    idx_in_list = selected_list.index(c)
                    selected_list[idx_in_list] = c_prime
                    selected_set.remove(c)
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
                        # Revert if no improvement (keep SA simple: only accept improvements or safe moves)
                        # Actually for SA we might want to accept non-improving moves too, but let's keep it simple first
                        pass
                    
                    break
                else:
                    # Check if c_prime covers all risky points
                    if np.all(np.isin(risky_points, neighbors[c_prime])):
                        # Safe swap
                        coverage_count[neighbors[c]] -= 1
                        coverage_count[neighbors[c_prime]] += 1
                        
                        idx_in_list = selected_list.index(c)
                        selected_list[idx_in_list] = c_prime
                        selected_set.remove(c)
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
                            pass
                        break
            else:
                # No valid swap found for this c, try a different c
                continue
            
            # Cool down
            T = max(T * cooling_rate, T_min)
        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0
    
    # Run a few greedy instances
    best_code = None
    best_len = float('inf')
    
    # Run 5 greedy instances (faster now with incremental update)
    for seed in range(5):
        if time.time() > deadline - 80: # Leave 80s for local search
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
