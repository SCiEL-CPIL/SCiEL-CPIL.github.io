import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
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
    # We will store these as lists of indices, and also as bitmasks
    neighbors = [None] * total_points
    masks = [0] * total_points
    
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        ball = [idx]
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
            ball.append(n_idx)
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            ball.append(n_idx)
            
        neighbors[idx] = ball
        # Create bitmask
        m = 0
        for p in ball:
            m |= (1 << p)
        masks[idx] = m

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def greedy_construction(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # uncovered_counts[b] = number of uncovered points in ball b
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        
        # Initial state: all uncovered
        # We need to pick codewords.
        # To speed up, we can just pick random until covered, then prune?
        # No, greedy is better.
        
        # Optimization: Use a heap or just scan? Scanning 6561 items is fast.
        # But we do this ~486 times. 6561 * 486 ~ 3M ops. Fine.
        
        while True:
            if not np.any(uncovered_counts > 0):
                break
            
            # Find max uncovered count
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            
            # Random tie-breaking
            candidates = np.where(uncovered_counts == max_count)[0]
            best_idx = rng.choice(candidates)
            
            selected.append(best_idx)
            
            # Update uncovered counts
            for p in neighbors[best_idx]:
                if not is_covered[p]:
                    is_covered[p] = True
                    # Decrement uncovered_counts for all balls containing p
                    # This is expensive if done naively.
                    # We can precompute reverse neighbors?
                    # rev_neighbors[p] is list of b such that p in ball(b).
                    # Size of rev_neighbors[p] is 17.
                    # So 17 * 17 = 289 updates per codeword.
                    # 486 * 289 ~ 140k ops. Very fast.
                    pass # We need rev_neighbors

        return selected

    # Precompute reverse neighbors for greedy
    rev_neighbors = [[] for _ in range(total_points)]
    for b in range(total_points):
        for p in neighbors[b]:
            rev_neighbors[p].append(b)

    def fast_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        
        while True:
            if not np.any(uncovered_counts > 0):
                break
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            best_idx = rng.choice(candidates)
            selected.append(best_idx)
            
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
            for i in range(len(selected_list)):
                idx = selected_list[i]
                # Check if any point in ball(idx) has count 1
                # If min(coverage_count[ball(idx)]) > 1, remove
                if np.min(coverage_count[neighbors[idx]]) > 1:
                    selected_list.pop(i)
                    coverage_count[neighbors[idx]] -= 1
                    changed = True
                    break
        return selected_list

    def local_search(selected_indices, deadline):
        selected_list = prune(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        
        # Build risky_mask
        # risky_mask bit i is 1 if coverage_count[i] == 1
        risky_mask = 0
        for i in range(total_points):
            if coverage_count[i] == 1:
                risky_mask |= (1 << i)
        
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(999)
        stall = 0
        max_stall = 10000
        
        while time.time() < deadline and stall < max_stall:
            # 1. Try to remove
            removed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                # Check if removal is safe: no point in ball(idx) is risky
                # mask[idx] & risky_mask == 0
                if (masks[idx] & risky_mask) == 0:
                    # Remove
                    selected_list.pop(i)
                    # Update coverage
                    for p in neighbors[idx]:
                        coverage_count[p] -= 1
                        if coverage_count[p] == 1:
                            risky_mask |= (1 << p)
                        elif coverage_count[p] == 0:
                            # Should not happen if check was correct
                            pass
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                    removed = True
                    stall = 0
                    break
            
            if removed:
                continue
                
            # 2. Try to swap
            if not selected_list:
                break
            
            c = selected_list[rng.integers(len(selected_list))]
            
            # Find valid swaps
            # We want to pick a c_prime such that (masks[c] & ~masks[c_prime] & risky_mask) == 0
            # This means all risky points covered by c must also be covered by c_prime.
            
            # To make it more effective, try neighbors of c first
            candidates = neighbors[c][1:]
            # Add some randoms
            for _ in range(5):
                candidates.append(rng.integers(total_points))
            
            swapped = False
            for c_prime in candidates:
                if c_prime == c:
                    continue
                if c_prime in selected_list: # Check set membership? List is small, but set is better
                    continue
                
                # Check validity
                if (masks[c] & ~masks[c_prime] & risky_mask) == 0:
                    # Perform swap
                    pos = selected_list.index(c)
                    selected_list[pos] = c_prime
                    
                    # Update coverage
                    for p in neighbors[c]:
                        coverage_count[p] -= 1
                        if coverage_count[p] == 1:
                            risky_mask |= (1 << p)
                        elif coverage_count[p] == 0:
                            # Invalid? No, we checked risky points.
                            # But points with count 2 become 1.
                            pass
                    for p in neighbors[c_prime]:
                        coverage_count[p] += 1
                        if coverage_count[p] == 2:
                            risky_mask &= ~(1 << p)
                        elif coverage_count[p] == 1:
                            # Becomes 2, so no longer risky
                            pass
                        # If it was 1, it becomes 2. Risky mask clears.
                        # If it was 0, it becomes 1. Risky mask sets.
                        # Wait, my update logic for risky_mask is tricky.
                        # Let's just recalculate risky_mask periodically or carefully.
                        
                    swapped = True
                    stall = 0
                    break
            
            if not swapped:
                stall += 1
                # If stalled, maybe try a random restart or 2-opt?
                # For now, just break if stalled too much?
                # Or try more random candidates.
        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0
    
    best_code = None
    best_len = float('inf')
    
    # Run greedy
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        selected = fast_greedy(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = fast_greedy(0)
        best_code = prune(best_code)
        best_len = len(best_code)
        
    # Local Search
    best_code = local_search(best_code, deadline)
    
    # Final prune
    best_code = prune(best_code)
    
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
