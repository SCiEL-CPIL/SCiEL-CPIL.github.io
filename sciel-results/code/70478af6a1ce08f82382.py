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

    def simulated_annealing_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        
        # Calculate initial uncovered
        uncovered_mask = (coverage_count == 0)
        uncovered_count = np.sum(uncovered_mask)
        
        best_len = len(selected_list)
        best_uncovered = uncovered_count
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        # SA Parameters
        # We want to minimize Size + Penalty * Uncovered
        # Start with high penalty to ensure coverage, then lower it
        # But since we start from a valid cover (uncovered=0), we can start with moderate penalty
        # to allow exploring "bad" covers that lead to smaller ones.
        
        T_start = 10.0
        T_end = 0.1
        alpha_start = 5.0
        alpha_end = 0.0
        
        # Number of steps
        steps = 50000
        step = 0
        
        while time.time() < deadline and step < steps:
            # Cooling
            progress = step / steps
            T = T_start * (T_end / T_start) ** progress
            alpha = alpha_start * (1 - progress)
            
            # Objective function: Size + alpha * Uncovered
            # Current cost
            current_cost = len(selected_set) + alpha * uncovered_count
            
            # Generate a move
            # 1. Remove a random codeword
            # 2. Add a random codeword
            # 3. Swap
            
            move_type = rng.integers(3)
            
            if len(selected_set) == 0:
                # Must add
                c_add = rng.integers(total_points)
                # Tentatively add
                old_uncovered = uncovered_count
                # Update coverage
                # We need to check if this reduces uncovered
                # Actually, let's just do the update and calculate delta
                pass # Handled below in generic logic if we restructure
            
            if move_type == 0 and len(selected_set) > 0:
                # Remove
                c_remove = list(selected_set)[rng.integers(len(selected_set))]
                # Tentative removal
                # Points that were covered ONLY by c_remove become uncovered
                # We need to know which points have coverage == 1 and are in ball(c_remove)
                
                # Efficient check:
                # old_cov = coverage_count
                # new_cov = old_cov - 1 for neighbors[c_remove]
                # delta_uncovered = sum(new_cov[neighbors[c_remove]] == 0) - sum(old_cov[neighbors[c_remove]] == 0)
                # Since old_cov >= 0, and we are removing, old_cov[neighbors] can be 0 or >0.
                # If old_cov[p] == 0, it was uncovered. If we remove c, it's still uncovered.
                # If old_cov[p] == 1, it was covered. If we remove c, it becomes uncovered.
                # So delta_uncovered = count of p in ball(c_remove) where old_cov[p] == 1.
                
                ball = neighbors[c_remove]
                delta_uncovered = np.sum(coverage_count[ball] == 1)
                
                # Update coverage
                coverage_count[ball] -= 1
                uncovered_count += delta_uncovered
                
                new_cost = len(selected_set) - 1 + alpha * uncovered_count
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or (rng.random() < np.exp(-delta_cost / T) if T > 0 else False):
                    # Accept
                    selected_set.discard(c_remove)
                    # No need to update best if we track at the end or when valid
                    if uncovered_count == 0:
                        if len(selected_set) < best_len:
                            best_len = len(selected_set)
                            best_code = list(selected_set)
                else:
                    # Revert
                    coverage_count[ball] += 1
                    uncovered_count -= delta_uncovered
                    
            elif move_type == 1:
                # Add
                # Pick a random candidate. To be more effective, pick one that covers many uncovered points.
                # But for SA, random is okay, or we can bias.
                # Let's pick a random index.
                c_add = rng.integers(total_points)
                
                if c_add in selected_set:
                    continue # Skip if already in set
                
                ball = neighbors[c_add]
                # delta_uncovered: points that were uncovered (cov==0) and are in ball(c_add)
                # When we add c_add, cov becomes 1 for those points.
                # So they become covered.
                # delta_uncovered = - count of p in ball(c_add) where old_cov[p] == 0
                
                # Note: If cov[p] > 0, it stays covered. If cov[p] == 0, it becomes covered.
                # So uncovered_count decreases by count(cov==0 in ball).
                
                delta_uncovered = -np.sum(coverage_count[ball] == 0)
                
                coverage_count[ball] += 1
                uncovered_count += delta_uncovered
                
                new_cost = len(selected_set) + 1 + alpha * uncovered_count
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or (rng.random() < np.exp(-delta_cost / T) if T > 0 else False):
                    # Accept
                    selected_set.add(c_add)
                    if uncovered_count == 0:
                        if len(selected_set) < best_len:
                            best_len = len(selected_set)
                            best_code = list(selected_set)
                else:
                    # Revert
                    coverage_count[ball] -= 1
                    uncovered_count -= delta_uncovered
                    
            else:
                # Swap
                if len(selected_set) == 0:
                    continue
                
                c_remove = list(selected_set)[rng.integers(len(selected_set))]
                c_add = rng.integers(total_points)
                
                if c_add == c_remove or c_add in selected_set:
                    continue
                
                ball_rem = neighbors[c_remove]
                ball_add = neighbors[c_add]
                
                # Calculate delta_uncovered
                # Points in ball_rem:
                # If cov==1, they become uncovered (unless also in ball_add and cov becomes 1+1-1=1? No, order matters)
                # Let's calculate new_cov for all affected points.
                # Affected points = union(ball_rem, ball_add)
                
                # To be efficient, we can just apply changes and check.
                # But we need delta for acceptance.
                
                # Let's compute the change in uncovered count directly.
                # Old state: cov
                # New state: cov - 1(ball_rem) + 1(ball_add)
                
                # We can iterate over the union of balls if they are small, but 17+17 is small.
                # However, vectorized is faster.
                
                # Let's just do the tentative update, calculate new uncovered count, and compare.
                # This requires calculating `np.sum(new_cov == 0)`.
                # To avoid full array sum, we can just check the affected points.
                
                # Get unique affected points
                affected = np.unique(np.concatenate([ball_rem, ball_add]))
                
                old_cov_affected = coverage_count[affected].copy()
                
                # Apply changes
                coverage_count[ball_rem] -= 1
                coverage_count[ball_add] += 1
                
                new_cov_affected = coverage_count[affected]
                
                old_uncov_affected = np.sum(old_cov_affected == 0)
                new_uncov_affected = np.sum(new_cov_affected == 0)
                
                delta_uncovered = new_uncov_affected - old_uncov_affected
                uncovered_count += delta_uncovered
                
                new_cost = len(selected_set) + alpha * uncovered_cost # Size is same
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or (rng.random() < np.exp(-delta_cost / T) if T > 0 else False):
                    # Accept
                    selected_set.discard(c_remove)
                    selected_set.add(c_add)
                    if uncovered_count == 0:
                        if len(selected_set) < best_len:
                            best_len = len(selected_set)
                            best_code = list(selected_set)
                else:
                    # Revert
                    coverage_count[ball_add] -= 1
                    coverage_count[ball_rem] += 1
                    uncovered_count -= delta_uncovered
            
            step += 1
            
            # Periodic pruning?
            # Pruning can be expensive. Maybe do it every 1000 steps if valid.
            if step % 1000 == 0 and uncovered_count == 0:
                pruned = prune(list(selected_set))
                if len(pruned) < len(selected_set):
                    # Rebuild state
                    selected_set = set(pruned)
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_set:
                        coverage_count[neighbors[idx]] += 1
                    uncovered_count = np.sum(coverage_count == 0)
                    if len(selected_set) < best_len:
                        best_len = len(selected_set)
                        best_code = list(selected_set)
        
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
    
    # Local Search / SA
    best_code = simulated_annealing_optimize(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
