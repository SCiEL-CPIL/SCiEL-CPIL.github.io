"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    """
    Greedy set-cover for covering code K_q(n, radius) with local search polish.
    Uses numpy arrays for efficient coverage tracking and a swap-based local search
    to reduce codeword count beyond simple removal-based polish.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1
    
    def index_to_word(idx, q, n):
        """Convert a linear index back to a word (tuple of ints)."""
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)
    
    # Precompute ball indices for all possible codewords
    # For radius=1: ball = {word} ∪ {word with one position changed to each other value}
    # Build ball_arrays as int64 arrays
    ball_arrays = []
    for idx in range(total_words):
        word = index_to_word(idx, q, n)
        covered = [idx]
        for pos in range(n):
            digit = word[pos]
            weight = q ** (n - 1 - pos)
            for val in range(q):
                if val != digit:
                    new_idx = idx + (val - digit) * weight
                    covered.append(new_idx)
        ball_arrays.append(np.array(covered, dtype=np.int64))
    
    # Greedy set cover
    uncovered = np.ones(total_words, dtype=np.uint8)
    codeword_indices = []
    
    while uncovered.any():
        if time.time() > deadline - 10:
            break
        
        # Find the codeword that covers the most uncovered words
        best_count = -1
        best_idx = -1
        
        for idx in range(total_words):
            ball = ball_arrays[idx]
            count = np.count_nonzero(uncovered[ball])
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            break
        
        codeword_indices.append(best_idx)
        uncovered[ball_arrays[best_idx]] = 0
    
    # Check validity
    if uncovered.any():
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base_solution
    
    # Local search polish: try to remove codewords and swap codewords
    # First pass: try removing each codeword (if its coverage is redundant)
    m = len(codeword_indices)
    
    # Precompute coverage counts: for each word, how many codewords cover it
    cover_count = np.zeros(total_words, dtype=np.int32)
    for ci in codeword_indices:
        cover_count[ball_arrays[ci]] += 1
    
    # Try removing codewords whose ball is fully covered by others
    changed = True
    while changed and time.time() < deadline - 15:
        changed = False
        for i in range(len(codeword_indices)):
            ci = codeword_indices[i]
            ball = ball_arrays[ci]
            would_uncover = (cover_count[ball] == 1)
            if not would_uncover.any():
                cover_count[ball] -= 1
                codeword_indices.pop(i)
                changed = True
                break  # restart from beginning
    
    # Second pass: swap-based local search
    # Try swapping a codeword with another that covers all uniquely-covered words
    # and gains additional coverage
    if time.time() < deadline - 5:
        m = len(codeword_indices)
        # Rebuild cover_count after removals
        cover_count = np.zeros(total_words, dtype=np.int32)
        for ci in codeword_indices:
            cover_count[ball_arrays[ci]] += 1
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(codeword_indices)):
                ci = codeword_indices[i]
                ball_i = ball_arrays[ci]
                unique_to_i = (cover_count[ball_i] == 1)
                
                if time.time() > deadline - 10:
                    break
                
                # Try swapping with a candidate not in current set
                # Use random candidates for speed
                max_tries = 100
                for attempt in range(max_tries):
                    cand = np.random.randint(0, total_words)
                    if cand in codeword_indices:
                        continue
                    
                    ball_cand = ball_arrays[cand]
                    
                    # Check if swap is valid: after removing ci and adding cand,
                    # no word should have coverage count 0
                    cover_count[ball_i] -= 1
                    cover_count[ball_cand] += 1
                    if np.any(cover_count == 0):
                        cover_count[ball_cand] -= 1
                        cover_count[ball_i] += 1
                        continue
                    # Valid swap - accept it to rearrange coverage
                    codeword_indices[i] = cand
                    improved = True
                    break
                
                if improved:
                    break
    
    # Final removal pass after swaps
    if time.time() < deadline - 3:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for ci in codeword_indices:
            cover_count[ball_arrays[ci]] += 1
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(codeword_indices)):
                ci = codeword_indices[i]
                ball = ball_arrays[ci]
                if not (cover_count[ball] == 1).any():
                    cover_count[ball] -= 1
                    codeword_indices.pop(i)
                    changed = True
                    break
    
    # Convert to list of lists
    codewords = [list(index_to_word(ci, q, n)) for ci in codeword_indices]
    
    # Final validity check
    final_cover = np.zeros(total_words, dtype=np.uint8)
    for ci in codeword_indices:
        final_cover[ball_arrays[ci]] = 1
    
    if final_cover.any() == 0 or np.any(final_cover == 0):
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base_solution
    
    return codewords

# EVOLVE-BLOCK-END
