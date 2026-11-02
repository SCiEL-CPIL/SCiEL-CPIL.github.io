"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    """
    Greedy set-cover for covering code K_q(n, radius).
    Uses bitset representation for efficient coverage tracking.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    total_words = q ** n
    
    # For radius=1, each codeword covers 1 + n*(q-1) words
    # Precompute ball indices for all possible codewords
    # Ball index: for a word w, the indices it covers are:
    #   - w itself
    #   - w with one position changed to each other value
    
    def word_to_index(word):
        """Convert a word (tuple of ints) to a linear index in 0..q^n-1."""
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx
    
    def index_to_word(idx, q, n):
        """Convert a linear index back to a word (tuple of ints)."""
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)
    
    # Precompute for each word index, the list of indices in its ball of radius `radius`
    # For radius=1: ball = {word} ∪ {word with one position changed}
    ball_indices = []
    for idx in range(total_words):
        word = index_to_word(idx, q, n)
        covered = set()
        covered.add(idx)
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    covered.add(word_to_index(tuple(new_word)))
        ball_indices.append(covered)
    
    # Greedy set cover:
    # uncovered: numpy array of bool, True if word is not yet covered
    # For each candidate codeword, track how many uncovered words it covers
    uncovered = np.ones(total_words, dtype=np.bool_)
    codewords = []
    
    # Precompute ball as list of arrays for faster numpy operations
    ball_arrays = [np.array(sorted(b), dtype=np.int64) for b in ball_indices]
    
    while uncovered.any():
        if time.time() > deadline - 5:
            # Not enough time to guarantee valid result; break and use fallback
            break
        
        # Find the codeword that covers the most uncovered words
        best_count = -1
        best_idx = -1
        
        # Check all potential codewords
        for idx in range(total_words):
            # Quick upper bound: can't cover more than remaining uncovered in its ball
            ball = ball_arrays[idx]
            count = np.count_nonzero(uncovered[ball])
            if count > best_count:
                best_count = count
                best_idx = idx
                if count == len(ball):
                    # Perfect: covers all words in its ball that are uncovered
                    # Can't do better than this for this codeword, but another might cover more
                    pass
        
        if best_idx == -1:
            break
        
        # Add this codeword
        word = index_to_word(best_idx, q, n)
        codewords.append(list(word))
        
        # Update uncovered
        uncovered[ball_arrays[best_idx]] = False
    
    # Check if we have a valid cover
    if uncovered.any():
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base_solution
    
    return codewords

# EVOLVE-BLOCK-END
