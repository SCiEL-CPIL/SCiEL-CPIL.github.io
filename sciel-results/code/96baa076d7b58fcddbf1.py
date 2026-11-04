"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def construct(q, n, radius):
    """
    Greedy set-covering with simulated annealing refinement for ternary covering code.
    """
    start_time = time.time()
    time_limit = 115  # leave some buffer before 120s timeout
    
    # Total number of words in the space
    total_words = q ** n
    
    # Helper functions for index conversion
    def tuple_to_idx(t):
        idx = 0
        for i, val in enumerate(t):
            idx = idx * q + val
        return idx

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute all words as tuples
    all_words_tuples = [idx_to_tuple(i) for i in range(total_words)]
    
    # Precompute balls as lists of indices
    ball_indices = []
    for c in range(total_words):
        center_t = all_words_tuples[c]
        ball = [c]
        for r in range(1, radius + 1):
            if r == 1:
                for pos in range(n):
                    orig = center_t[pos]
                    for new_val in range(q):
                        if new_val == orig:
                            continue
                        new_t = list(center_t)
                        new_t[pos] = new_val
                        ball.append(tuple_to_idx(tuple(new_t)))
        ball_indices.append(ball)
        
    # Greedy Set Cover
    uncovered = set(range(total_words))
    centers = []
    
    while uncovered:
        if time.time() - start_time > time_limit * 0.2:
            break
        best_center = -1
        best_cov = -1
        for c in range(total_words):
            cov = 0
            for w in ball_indices[c]:
                if w in uncovered:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_center = c
        
        if best_center == -1 or best_cov == 0:
            break
            
        centers.append(best_center)
        for w in ball_indices[best_center]:
            if w in uncovered:
                uncovered.remove(w)
                
    # If greedy didn't finish, fill rest
    if uncovered:
        for w in list(uncovered):
            centers.append(w)
            uncovered.discard(w)

    # Simulated Annealing
    current_centers = centers[:]
    coverage_counts = np.zeros(total_words, dtype=np.int32)
    
    for c in current_centers:
        for w in ball_indices[c]:
            coverage_counts[w] += 1
            
    best_centers = current_centers[:]
    best_count = len(current_centers)
    curr_uncovered = 0
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    random.seed(42)
    
    while T > T_min and time.time() - start_time < time_limit:
        if len(current_centers) < 2:
            break
            
        # Pick a random center to modify
        idx = random.randint(0, len(current_centers) - 1)
        c = current_centers[idx]
        
        # Move 1: Remove (70%)
        # Move 2: Swap (30%)
        
        if random.random() < 0.7:
            # Try Remove
            for w in ball_indices[c]:
                coverage_counts[w] -= 1
                
            new_uncovered = np.sum(coverage_counts == 0)
            
            # Cost function: Count + 100 * Uncovered
            old_cost = len(current_centers) + 100 * curr_uncovered
            new_cost = (len(current_centers) - 1) + 100 * new_uncovered
            
            delta = new_cost - old_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_centers.pop(idx)
                curr_uncovered = new_uncovered
                if new_uncovered == 0 and len(current_centers) < best_count:
                    best_count = len(current_centers)
                    best_centers = current_centers[:]
            else:
                # Reject
                for w in ball_indices[c]:
                    coverage_counts[w] += 1
        else:
            # Try Swap
            c_new = random.randint(0, total_words - 1)
            if c == c_new:
                T *= cooling
                continue
                
            for w in ball_indices[c]:
                coverage_counts[w] -= 1
            for w in ball_indices[c_new]:
                coverage_counts[w] += 1
                
            new_uncovered = np.sum(coverage_counts == 0)
            
            old_cost = len(current_centers) + 100 * curr_uncovered
            new_cost = len(current_centers) + 100 * new_uncovered
            
            delta = new_cost - old_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_centers[idx] = c_new
                curr_uncovered = new_uncovered
            else:
                # Reject
                for w in ball_indices[c_new]:
                    coverage_counts[w] -= 1
                for w in ball_indices[c]:
                    coverage_counts[w] += 1
        
        T *= cooling

    # Final verification
    final_centers = best_centers
    # Ensure validity
    check_counts = np.zeros(total_words, dtype=np.int32)
    for c in final_centers:
        for w in ball_indices[c]:
            check_counts[w] += 1
    if np.any(check_counts == 0):
        # Fallback to greedy
        final_centers = centers
        
    return [list(all_words_tuples[c]) for c in final_centers]
# EVOLVE-BLOCK-END
