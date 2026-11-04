"""Improved program: greedy set cover with bitsets for K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random


def _ball_mask(q, n, radius, center_int):
    """Return an int bitmask of all words within Hamming distance radius of center_int."""
    center = []
    for i in range(n):
        center.append((center_int // (q ** (n - 1 - i))) % q)
    
    mask = 1 << center_int
    
    # Precompute combinations of positions for each distance
    # For radius 1, we just change one position
    # For general radius, we change up to radius positions
    
    if radius == 1:
        for pos in range(n):
            orig = center[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                word = center.copy()
                word[pos] = new_val
                # Convert back to int
                w_int = 0
                for j in range(n):
                    w_int = w_int * q + word[j]
                mask |= (1 << w_int)
    else:
        # General case: iterate over all combinations of positions to change
        from itertools import combinations
        for d in range(1, radius + 1):
            for positions in combinations(range(n), d):
                # For each combination of positions, try all non-identity assignments
                for new_vals in itertools.product(range(q), repeat=d):
                    if all(new_vals[i] == center[positions[i]] for i in range(d)):
                        continue
                    word = center.copy()
                    for i, pos in enumerate(positions):
                        word[pos] = new_vals[i]
                    w_int = 0
                    for j in range(n):
                        w_int = w_int * q + word[j]
                    mask |= (1 << w_int)
    
    return mask


def _word_to_int(word, q):
    """Convert a list of digits to an integer in base q."""
    val = 0
    for d in word:
        val = val * q + d
    return val


def _int_to_word(w_int, q, n):
    """Convert an integer to a list of n digits in base q."""
    word = []
    for i in range(n - 1, -1, -1):
        word.append(w_int % q)
        w_int //= q
    return word


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin from 120s limit
    
    total_words = q ** n
    
    # Precompute ball masks for all words
    # This is O(total_words * n * q * radius) approximately
    balls = [0] * total_words
    for i in range(total_words):
        balls[i] = _ball_mask(q, n, radius, i)
    
    # All bits set initially
    all_mask = (1 << total_words) - 1
    best_code = None
    best_size = float('inf')
    
    # Multiple restarts with different random orderings
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Try different strategies
        # Strategy 1: Greedy with random tie-breaking
        uncovered = all_mask
        code = []
        
        # Create a shuffled list of candidates
        candidates = list(range(total_words))
        rng.shuffle(candidates)
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_c = -1
            best_count = 0
            
            # To speed up, we can sample or iterate through all
            # For 729 words, iterating through all is fine
            for c in candidates:
                count = bin(uncovered & balls[c]).count('1')
                if count > best_count:
                    best_count = count
                    best_c = c
                    if best_count == total_words:
                        break
            
            if best_c == -1 or best_count == 0:
                break
            
            code.append(best_c)
            uncovered &= ~balls[best_c]
        
        if uncovered == 0 and len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
        
        # If we found a very good code, we can break early
        if best_size <= 73:
            break
    
    # If we didn't find any valid code, fall back to the tetracode construction
    if best_code is None:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # For small n, just return all words
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Convert integer codes back to word lists
    result = [_int_to_word(c, q, n) for c in best_code]
    return result
# EVOLVE-BLOCK-END
