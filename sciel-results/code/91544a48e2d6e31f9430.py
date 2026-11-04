"""Rewritten program using Greedy Set Cover with local search for ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Helper to convert index (0..6560) to list of digits (base 3, length 8)
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Pre-compute the coverage sets for every possible codeword index.
    coverage_sets = np.empty((total_points, ball_size), dtype=np.int32)
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Precompute digit values for all indices to avoid repeated division
    # digits[idx, pos] = digit at position pos for index idx
    digits = np.zeros((total_points, n), dtype=np.int8)
    for idx in range(total_points):
        temp = idx
        for pos in range(n):
            digits[idx, pos] = temp % q
            temp //= q

    for idx in range(total_points):
        coverage_sets[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            original_digit = digits[idx, pos]
            power = powers[pos]
            
            # Try adding 1
            new_digit = (original_digit + 1) % q
            if new_digit == 0:
                delta_idx = -2 * power
            else:
                delta_idx = power
            coverage_sets[idx, neighbor_idx] = idx + delta_idx
            neighbor_idx += 1
            
            # Try subtracting 1
            new_digit = (original_digit - 1) % q
            if new_digit == 2:
                delta_idx = 2 * power
            else:
                delta_idx = -power
            coverage_sets[idx, neighbor_idx] = idx + delta_idx
            neighbor_idx += 1

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Greedy Set Cover
    is_covered = np.zeros(total_points, dtype=bool)
    selected_indices = []
    
    while True:
        if time.time() > deadline - 20.0:  # Reserve 20s for local search
            break
            
        uncovered_counts = np.sum(~is_covered[coverage_sets], axis=1)
        best_idx = np.argmax(uncovered_counts)
        best_count = uncovered_counts[best_idx]
        
        if best_count == 0:
            break
            
        selected_indices.append(best_idx)
        is_covered[coverage_sets[best_idx]] = True
        
        if np.all(is_covered):
            break

    # Pruning pass: remove redundant codewords
    if len(selected_indices) > 1:
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_indices:
            coverage_count[coverage_sets[idx]] += 1

        unique_counts = np.zeros(len(selected_indices), dtype=np.int32)
        for i, idx in enumerate(selected_indices):
            ball = coverage_sets[idx]
            unique_counts[i] = np.sum(coverage_count[ball] == 1)

        changed = True
        while changed and time.time() < deadline - 15.0:
            changed = False
            redundant_indices = np.where(unique_counts == 0)[0]
            if len(redundant_indices) == 0:
                break
            rem_idx = redundant_indices[0]
            rem_codeword = selected_indices[rem_idx]
            ball = coverage_sets[rem_codeword]
            coverage_count[ball] -= 1
            selected_indices.pop(rem_idx)
            unique_counts = np.zeros(len(selected_indices), dtype=np.int32)
            for i, idx in enumerate(selected_indices):
                ball_i = coverage_sets[idx]
                unique_counts[i] = np.sum(coverage_count[ball_i] == 1)
            changed = True

    # Local search: try to reduce the number of codewords
    # We can try to replace a codeword with a different one that covers the same unique points
    # or try to remove a codeword and re-cover its unique points with other codewords
    
    # A more effective approach: iteratively try to remove a codeword and see if the rest can cover
    # the points it uniquely covered by adding at most one new codeword.
    
    # For now, let's do a simple local search:
    # 1. Try removing each codeword one by one.
    # 2. If the code remains valid, keep the removal.
    # 3. If not, try to find a single codeword that covers all the newly uncovered points.
    
    # To make this efficient, we track the coverage count.
    if len(selected_indices) > 0:
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_indices:
            coverage_count[coverage_sets[idx]] += 1
        
        # Sort selected_indices by the number of unique points they cover (ascending)
        # to try removing the least "important" ones first.
        unique_counts = np.zeros(len(selected_indices), dtype=np.int32)
        for i, idx in enumerate(selected_indices):
            ball = coverage_sets[idx]
            unique_counts[i] = np.sum(coverage_count[ball] == 1)
        
        # We'll iterate through the sorted indices
        order = np.argsort(unique_counts)
        
        for i in order:
            if time.time() > deadline - 10.0:
                break
            
            idx = selected_indices[i]
            ball = coverage_sets[idx]
            
            # Check if any point in this ball has coverage_count == 1
            if np.any(coverage_count[ball] == 1):
                continue
            
            # This codeword is redundant
            coverage_count[ball] -= 1
            selected_indices.pop(i)
            # Note: indices in 'order' are now stale, but since we're only removing,
            # and we're iterating in a specific order, this is tricky.
            # Let's restart the loop or handle indices carefully.
            # For simplicity, let's just break and restart if we remove one.
            # This is less efficient but safer.
            break

    # Convert indices back to list of lists of integers
    result = []
    for idx in selected_indices:
        result.append(idx_to_word(idx))

    return result

# EVOLVE-BLOCK-END
