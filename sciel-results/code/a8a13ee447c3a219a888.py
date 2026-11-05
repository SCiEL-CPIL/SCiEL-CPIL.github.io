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
        uncovered_count = total_points
        
        while uncovered_count > 0:
            # Count uncovered points in each ball
            uncovered_mask = ~is_covered
            counts = np.sum(uncovered_mask[balls], axis=1)
            
            # Pick index with max count
            max_count = counts.max()
            if max_count == 0:
                break
            
            # Random tie-breaking
            candidates = np.where(counts == max_count)[0]
            best_idx = int(rng.choice(candidates))
            
            selected.append(best_idx)
            newly_covered = balls[best_idx]
            # Count how many were actually newly covered
            newly_covered_mask = newly_covered[~is_covered[newly_covered]]
            if len(newly_covered_mask) > 0:
                uncovered_count -= len(newly_covered_mask)
                is_covered[newly_covered] = True
            
        return selected

    def prune(selected, counts):
        """
        Remove redundant codewords.
        Returns new selected list and updated counts.
        """
        if not selected:
            return selected, counts
        
        # Convert to set for fast membership check
        selected_set = set(selected)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                c = selected[i]
                # If min count in this ball is > 1, it's redundant
                if counts[balls[c]].min() > 1:
                    counts[balls[c]] -= 1
                    selected.pop(i)
                    selected_set.discard(c)
                    changed = True
                    break # Restart scan
        return selected, counts

    def local_search(selected, counts, deadline):
        """
        Hill climbing with informed swaps and periodic full sweeps.
        """
        best_len = len(selected)
        best_selected = selected[:]
        best_counts = counts.copy()
        
        rng = random.Random()
        selected_set = set(selected)
        
        # Pre-compute for each codeword which other codewords' balls overlap with it
        # This helps identify "critical" points that are only covered by one codeword
        
        while time.time() < deadline:
            # 1. Try to prune
            selected, counts = prune(selected, counts)
            selected_set = set(selected)
            if len(selected) < best_len:
                best_len = len(selected)
                best_selected = selected[:]
                best_counts = counts.copy()
            
            if len(selected) == 0:
                break
            
            # 2. Try multiple random swaps per iteration
            num_swaps = min(50, max(1, int((deadline - time.time()) * 100)))
            for _ in range(num_swaps):
                if time.time() >= deadline:
                    break
                
                # Pick a random codeword
                i = rng.randint(0, len(selected) - 1)
                c = selected[i]
                
                # Pick a random neighbor
                nbrs = neighbors[c]
                c_prime = int(rng.choice(nbrs))
                
                # Check if c_prime is already in selected
                if c_prime in selected_set:
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
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    selected[i] = c_prime
                    # Try to prune immediately after a successful swap
                    selected, counts = prune(selected, counts)
                    selected_set = set(selected)
                    if len(selected) < best_len:
                        best_len = len(selected)
                        best_selected = selected[:]
                        best_counts = counts.copy()

        # Final polish: one more prune pass
        best_selected, best_counts = prune(best_selected, best_counts)
        
        return best_selected

    start_time = time.time()
    # Leave 5 seconds for final processing and safety
    deadline = start_time + 115.0
    
    best_code = None
    best_len = float('inf')
    
    # Strategy: First run gets more time for local search.
    # Subsequent runs get less time but can still improve.
    # Allocate 60% to first run, 25% to second, 15% to third.
    
    time_allocations = [0.60, 0.25, 0.15]
    total_alloc = sum(time_allocations)
    time_per_run = [(deadline - start_time) * alloc / total_alloc for alloc in time_allocations]
    
    for run_idx in range(3):
        if time.time() >= deadline:
            break
        
        run_deadline = min(start_time + sum(time_per_run[:run_idx + 1]), deadline)
        
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
            best_code = selected[:]
            
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

    # Final verification and safety check
    if time.time() < deadline - 3.0:
        # Do one final prune pass on the best code
        final_counts = np.zeros(total_points, dtype=np.int32)
        for idx in best_code:
            final_counts[balls[idx]] += 1
        best_code, final_counts = prune(best_code, final_counts)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
