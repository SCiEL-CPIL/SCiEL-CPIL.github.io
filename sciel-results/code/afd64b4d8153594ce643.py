"""Ternary covering code K_3(8,1) via greedy set cover with vectorized incidence."""
import time
import numpy as np
import itertools

# Deadline: evaluator kills at 150s; keep a safety margin.
DEADLINE = 135.0  # seconds from start


def construct(q, n, radius):
    start_time = time.time()
    
    # For the specific task: q=3, n=8, radius=1
    # Total words: 3^8 = 6561
    # Each codeword covers 1 + 8*2 = 17 words (itself + 8 positions * 2 alternatives)
    # Lower bound: ceil(6561/17) = 386
    # Best known: 486
    
    # Strategy: Greedy set cover
    # 1. Generate all 6561 words as integers 0..6560
    # 2. For each word, precompute the set of words within Hamming distance <= radius
    # 3. Greedily pick the word that covers the most uncovered words
    
    # Represent words as integers in base q
    total = q ** n
    
    # Convert int to word (list of n digits)
    def int_to_word(x):
        w = [0] * n
        for i in range(n):
            w[i] = x % q
            x //= q
        return w
    
    # Convert word to int
    def word_to_int(w):
        x = 0
        for i in range(n):
            x = x * q + w[i]
        return x
    
    # Precompute: for each word index, the list of word indices within Hamming distance <= radius
    # This is a 6561 x 6561 boolean matrix, but we'll store it as lists for efficiency
    
    # Actually, let's use a different approach:
    # For each word, generate all neighbors within Hamming distance <= radius
    # Store as a list of lists
    
    # But 6561 * 17 = ~111k entries, which is fine.
    
    # Let's precompute the neighbor lists
    # For each word w (as int), compute all words within Hamming distance <= radius
    
    # More efficient: for each position and each alternative value, compute the new int
    # For radius=1: neighbors = {w} U {w with one position changed}
    
    # Precompute base-q powers
    base_powers = [1] * n
    for i in range(1, n):
        base_powers[i] = base_powers[i-1] * q
    
    # For each word index x, compute the set of covered word indices
    # We'll build an array of sets (or lists)
    
    # Actually, for greedy set cover, we need to:
    # 1. Start with all words uncovered
    # 2. At each step, pick the codeword that covers the most uncovered words
    # 3. Mark those words as covered
    # 4. Repeat until all covered
    
    # To make this efficient with numpy:
    # - Maintain a boolean array 'covered' of size total
    # - For each candidate codeword, compute how many uncovered words it covers
    # - Pick the max
    
    # But checking all 6561 candidates at each step is expensive.
    # Alternative: use the fact that the covering is local.
    
    # Let's try a simpler greedy: 
    # Precompute for each word its neighborhood (list of word indices within distance <= radius)
    # Then do greedy set cover.
    
    # Precompute neighborhoods
    # For radius=1, n=8, q=3: each word has 1 + 8*2 = 17 neighbors
    
    neighborhoods = []
    for x in range(total):
        w = int_to_word(x)
        covered_set = [x]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                covered_set.append(word_to_int(w))
            w[pos] = orig
        neighborhoods.append(covered_set)
    
    # Convert to numpy arrays for faster operations
    # Actually, let's keep as lists and use a different approach.
    
    # Greedy set cover:
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # To speed up, we can use the following trick:
    # At each step, for each uncovered word, count how many candidate codewords cover it.
    # But that's expensive too.
    
    # Alternative: just iterate through all candidates and pick the one with max coverage.
    # With 6561 candidates and ~400 iterations, that's 2.6M operations, each involving
    # checking 17 neighbors. That's ~44M operations, which should be fine in 135s.
    
    # But let's vectorize where possible.
    
    # Precompute neighborhoods as a 2D array (with padding)
    max_neighbors = 1 + n * (q - 1)
    neighbor_array = np.full((total, max_neighbors), -1, dtype=np.int32)
    neighbor_counts = np.zeros(total, dtype=np.int32)
    
    for x in range(total):
        for i, y in enumerate(neighborhoods[x]):
            neighbor_array[x, i] = y
            neighbor_counts[x] = i + 1
    
    # Now greedy:
    while not covered.all():
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        
        # For each candidate codeword x, compute how many uncovered words it covers
        # covered_count[x] = sum over y in neighborhood[x] of (not covered[y])
        
        # Vectorized: for each x, take neighbor_array[x, :neighbor_counts[x]] and check covered
        # This is tricky to vectorize fully. Let's do it in a loop but with numpy for the inner part.
        
        best_x = -1
        best_count = -1
        
        for x in range(total):
            if covered[x]:
                # If x is already covered, it might still be a good codeword choice
                # Actually, we should consider all x, not just uncovered ones
                pass
            
            # Get the neighborhood of x
            nb = neighbor_array[x, :neighbor_counts[x]]
            # Count how many are not covered
            count = np.sum(~covered[nb])
            
            if count > best_count:
                best_count = count
                best_x = x
        
        if best_x == -1 or best_count <= 0:
            # Should not happen if not all covered
            break
        
        # Add best_x to codewords
        codewords.append(best_x)
        
        # Mark its neighborhood as covered
        nb = neighbor_array[best_x, :neighbor_counts[best_x]]
        covered[nb] = True
    
    # Check if we completed within time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return what we have, but we need to ensure validity
        # If we didn't complete, we need to add more codewords to ensure coverage
        # Let's add all remaining uncovered words' neighborhoods
        while not covered.all():
            # Find an uncovered word
            uncovered_idx = np.where(~covered)[0]
            if len(uncovered_idx) == 0:
                break
            x = uncovered_idx[0]
            if x not in codewords:
                codewords.append(x)
            nb = neighbor_array[x, :neighbor_counts[x]]
            covered[nb] = True
    
    # Convert codewords from int to list of digits
    result = []
    seen = set()
    for x in codewords:
        if x not in seen:
            seen.add(x)
            result.append(int_to_word(x))
    
    return result
