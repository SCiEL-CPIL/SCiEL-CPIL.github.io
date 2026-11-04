"""
Improved program for the ternary covering code problem K_3(7,1).
Uses a greedy set cover algorithm with bitset optimization.
"""

import time
import itertools

def construct(q, n, radius):
    # The problem is specifically for q=3, n=7, radius=1.
    # If parameters differ, we fall back to a simple construction to ensure validity.
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Direct sum with tetracode if n >= 4, else brute force small
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Small n, just pick everything or a simple grid
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin before 120s limit

    # Total number of words
    total_words = 3 ** 7
    words = [list(w) for w in itertools.product(range(3), repeat=7)]
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute balls (masks) for each word
    # ball_mask[i] is an integer with bits set for all words within distance 1 of words[i]
    ball_masks = [0] * total_words
    for i, w in enumerate(words):
        mask = 0
        # The word itself
        mask |= (1 << i)
        # All words at distance 1
        for pos in range(7):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w = w[:]
                new_w[pos] = new_val
                idx = word_to_idx[tuple(new_w)]
                mask |= (1 << idx)
        ball_masks[i] = mask

    # Initialize uncovered set as a bitmask with all bits set
    all_mask = (1 << total_words) - 1
    uncovered = all_mask
    
    code = []
    
    # Greedy set cover
    while uncovered:
        # Check deadline
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = 0
        
        # Find the word that covers the most uncovered words
        # We can optimize by skipping words that are already covered? 
        # No, a covered word can still cover other uncovered words.
        # However, if a word is covered, it doesn't matter if we pick it or not 
        # for its own coverage, but it might be the best center for others.
        
        # To speed up, we can iterate only through uncovered words? 
        # No, the optimal center might be a covered word. 
        # But usually, greedy picks uncovered words first. 
        # Let's just iterate all words for correctness and simplicity, 
        # as 2187 * bit_count is very fast in Python.
        
        for i in range(total_words):
            # Intersection of uncovered and ball of i
            # Python int bit_count is fast
            count = (uncovered & ball_masks[i]).bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
                # Early exit if we found a perfect cover for remaining? Unlikely but possible
                if best_count == uncovered.bit_count():
                    break
        
        if best_idx == -1:
            break # Should not happen if uncovered is not 0
            
        # Add to code
        code.append(words[best_idx])
        
        # Update uncovered
        uncovered &= ~ball_masks[best_idx]
        
        # Optional: Check if we are stuck in a loop or if time is running out
        # The loop terminates when uncovered becomes 0.

    # If we hit the deadline and didn't finish, we have a partial cover.
    # The problem requires a VALID cover. 
    # If we break early, we must ensure validity.
    # Since the greedy process is monotonic (uncovered only shrinks), 
    # if we stop early, some words remain uncovered.
    # To guarantee validity within the time limit, we should ensure we finish.
    # The greedy loop is ~200-250 iterations. Each iteration does 2187 bit ops.
    # 250 * 2187 * (cost of bit_count) is very small (< 0.1s).
    # So the deadline is likely never hit. 
    # But if it is, we can't return an invalid code.
    # We will assume it finishes. If not, we fall back to the 243 solution.
    
    # Verify validity just in case (slow, but only if we suspect issues)
    # Actually, let's just trust the greedy if it finishes.
    # If uncovered is not 0, we add the remaining uncovered words to the code to ensure validity.
    if uncovered:
        # Add all remaining uncovered words
        remaining_idx = 0
        while uncovered:
            # Get least significant set bit
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            code.append(words[idx])
            uncovered ^= lsb
            # Check deadline again
            if time.time() > deadline:
                 # If we are running out of time adding the rest, 
                 # we might timeout. But adding up to 2187 words is fast.
                 pass

    return code
