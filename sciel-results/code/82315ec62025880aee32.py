import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            # Simple tetracode based construction for q=3
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # For general n, just use product set if we don't have specific codes
            # But task is specific to 3,8,1
            pass
        # Fallback: Return all words (valid but large)
        return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute balls for all points
    # balls[idx] is an array of 17 indices: idx itself and its 16 neighbors
    balls = np.empty((total_points, ball_size), dtype=np.int32)
    neighbors = [None] * total_points
    
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        balls[idx, 0] = idx
        neighbor_idx = 1
        nbrs = []
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
            balls[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            nbrs.append(n_idx)
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            balls[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            nbrs.append(n_idx)
            
        neighbors[idx] = nbrs

    def idx_to_word(idx):
        w = [0] * n
        temp_idx = idx
        for i in range(n):
            w[i] = temp_idx % q
            temp_idx //= q
        return w

    def run_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Initial uncovered counts are all 1 (each point needs to be covered)
        # We pick the ball that covers the most uncovered points.
        
        while not np.all(is_covered):
            # Count uncovered points in each ball
            # ~is_covered is boolean mask of uncovered points
            uncovered_mask = ~is_covered
            # For each ball, count how many of its points are in uncovered_mask
            # balls is (6561, 17). uncovered_mask is (6561,).
            # We can sum uncovered_mask[balls] along axis 1
            counts = np.sum(uncovered_mask[balls], axis=1)
            
            # Pick index with max count
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Random tie-breaking
            candidates = np.where(counts == max_count)[0]
            best_idx = rng.choice(candidates)
            
            selected.append(best_idx)
            is_covered[balls[best_idx]] = True
            
        return selected

    def prune(selected, counts):
        """
        Remove redundant codewords.
        Returns new selected list and updated counts.
        """
        if not selected:
            return selected, counts
        
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to make popping easier? Or just break and restart.
            # Restarting is safer for correctness with indices.
            for i in range(len(selected)):
                c = selected[i]
                # If min count in this ball is > 1, it's redundant
                if np.min(counts[balls[c]]) > 1:
                    counts[balls[c]] -= 1
                    selected.pop(i)
                    changed = True
                    break # Restart scan
        return selected, counts

    def local_search(selected, counts, deadline):
        """
        Hill climbing with swaps.
        """
        best_len = len(selected)
        best_selected = selected[:]
        best_counts = counts.copy()
        
        rng = random.Random()
        
        while time.time() < deadline:
            # 1. Try to prune
            selected, counts = prune(selected, counts)
            if len(selected) < best_len:
                best_len = len(selected)
                best_selected = selected[:]
                best_counts = counts.copy()
            
            if len(selected) == 0:
                break
                
            # 2. Try a random swap
            # Pick a random codeword
            i = rng.randint(0, len(selected) - 1)
            c = selected[i]
            
            # Pick a random neighbor
            nbrs = neighbors[c]
            c_prime = rng.choice(nbrs)
            
            # Check if c_prime is already in selected
            # For small sets, linear search is fine. For 500, it's okay.
            # But we can use a set for faster lookup if needed.
            # Let's stick to list for now, or maintain a set in parallel.
            # Given the small size, `in` is acceptable.
            if c_prime in selected:
                continue
                
            # Tentative swap
            counts[balls[c]] -= 1
            counts[balls[c_prime]] += 1
            
            # Check validity:
            # Any point in balls[c] that was ONLY covered by c (count became 0)
            # must not exist.
            if np.any(counts[balls[c]] == 0):
                # Revert
                counts[balls[c_prime]] -= 1
                counts[balls[c]] += 1
            else:
                # Accept swap
                selected[i] = c_prime
                # Try to prune immediately after a successful swap
                selected, counts = prune(selected, counts)
                if len(selected) < best_len:
                    best_len = len(selected)
                    best_selected = selected[:]
                    best_counts = counts.copy()

        return best_selected

    start_time = time.time()
    # Leave 5 seconds for final processing and safety
    deadline = start_time + 115.0
    
    best_code = None
    best_len = float('inf')
    
    # Run multiple greedy instances
    # Each greedy instance is fast. Local search is the bottleneck.
    # Strategy: Run greedy, prune, then local search with remaining time.
    # If we run multiple greedy instances, we need to split time.
    
    # Let's try 3 runs of (Greedy + Prune + Local Search)
    # Or 1 Greedy + Long Local Search.
    # Given the landscape, multiple restarts might help escape local optima.
    
    num_runs = 3
    time_per_run = (deadline - start_time) / num_runs
    
    for run_idx in range(num_runs):
        run_deadline = start_time + (run_idx + 1) * time_per_run
        
        # Greedy
        seed = run_idx * 100 + 42
        selected = run_greedy(seed)
        
        # Initial Counts
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in selected:
            counts[balls[idx]] += 1
            
        # Prune
        selected, counts = prune(selected, counts)
        
        # Local Search
        selected = local_search(selected, counts, run_deadline)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
        if time.time() > deadline:
            break

    # Final safety: if best_code is None, do a quick greedy
    if best_code is None:
        selected = run_greedy(0)
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in selected:
            counts[balls[idx]] += 1
        selected, counts = prune(selected, counts)
        best_code = selected

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
