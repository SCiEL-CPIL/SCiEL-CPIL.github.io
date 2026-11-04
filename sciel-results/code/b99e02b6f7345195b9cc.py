"""Rewritten program using Greedy Set Cover with NumPy for ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    # We implement a general greedy solver for these parameters, 
    # but optimize heavily for q=3, n=8.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to simple construction if parameters differ, 
        # though the task contract implies specific values.
        # For safety, we return a valid (though potentially suboptimal) code.
        # The direct sum method works for n>=4.
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Brute force for small n
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    # Specific optimization for q=3, n=8, radius=1
    total_points = q ** n  # 6561
    
    # 1. Generate all possible codewords and map them to indices
    # We represent each word as an integer in base 3.
    # word_to_idx: dict or array. Since max idx is 6560, we can use an array or just compute index directly.
    # Let's create a list of all words as tuples for easy conversion, but we need to map word -> index.
    # Actually, we can just iterate 0 to total_points-1 and decode to get the word, or encode the word to get index.
    
    # Pre-compute the coverage sets for every possible codeword index.
    # coverage[i] will be an array of 17 indices (the word itself + 16 neighbors).
    
    # Helper to convert index (0..6560) to list of digits (base 3, length 8)
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Helper to get neighbors of a word (index)
    # We need to generate the 16 neighbors for each of the 6561 words.
    # This is 6561 * 16 operations.
    
    coverage_sets = np.empty((total_points, 1 + n * (q - 1)), dtype=np.int32)
    
    # Precompute powers of 3 for index calculation
    # index = d0 + d1*3 + d2*3^2 ...
    # To change digit i by delta, we add delta * 3^i.
    # We must handle wrap-around: if digit is 0 and delta is -1, it becomes 2 and we subtract 2*3^i.
    # If digit is 2 and delta is +1, it becomes 0 and we subtract 2*3^i.
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for idx in range(total_points):
        word = idx_to_word(idx)
        coverage_sets[idx, 0] = idx
        
        neighbor_idx = 1
        for pos in range(n):
            original_digit = word[pos]
            power = powers[pos]
            
            # Try adding 1
            new_digit = (original_digit + 1) % q
            # Calculate index change
            if new_digit == 0:
                # Wrapped from 2 to 0: subtract 2 * power
                delta_idx = -2 * power
            else:
                delta_idx = power
            coverage_sets[idx, neighbor_idx] = idx + delta_idx
            neighbor_idx += 1
            
            # Try subtracting 1
            new_digit = (original_digit - 1) % q
            if new_digit == 2:
                # Wrapped from 0 to 2: add 2 * power
                delta_idx = 2 * power
            else:
                delta_idx = -power
            coverage_sets[idx, neighbor_idx] = idx + delta_idx
            neighbor_idx += 1

    # 2. Greedy Set Cover
    is_covered = np.zeros(total_points, dtype=bool)
    selected_codewords = []
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    while True:
        if time.time() > deadline:
            break
            
        # Calculate how many uncovered points each candidate covers
        # is_covered[coverage_sets] gives a boolean matrix of shape (6561, 17)
        # ~is_covered[coverage_sets] gives True for uncovered points
        # sum along axis 1 gives the count of uncovered points for each candidate
        
        # This is the heavy operation, but vectorized.
        uncovered_counts = np.sum(~is_covered[coverage_sets], axis=1)
        
        best_idx = np.argmax(uncovered_counts)
        best_count = uncovered_counts[best_idx]
        
        if best_count == 0:
            break
            
        # Select this codeword
        selected_codewords.append(best_idx)
        
        # Mark its covered points as covered
        is_covered[coverage_sets[best_idx]] = True
        
        # Check if all covered
        if np.all(is_covered):
            break
            
        # Optional: early exit if we have too many, though greedy should be fine
        # If len(selected_codewords) > 1000: break # Just in case

    # Convert indices back to list of lists of integers
    result = []
    for idx in selected_codewords:
        result.append(idx_to_word(idx))
        
    return result

# EVOLVE-BLOCK-END
