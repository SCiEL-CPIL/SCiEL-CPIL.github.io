"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """Construct a large binary code of length 17 with minimum distance 6."""
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # Start with the first-order Reed-Muller code on 16 coordinates, padded to 17
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Bit 16 is 0 (padded)
            words.append(word)
    
    words_set = set(words)
    
    def hamming_dist(a, b):
        return (a ^ b).bit_count()
    
    def can_add(candidate, words_list):
        """Check if candidate can be added to the code."""
        for w in words_list:
            if hamming_dist(candidate, w) < MIN_DIST:
                return False
        return True
    
    # Greedy addition: try to add random candidates
    # Focus on candidates that differ from existing words in controlled ways
    
    # Strategy 1: Try all 2^17 = 131072 possible words, add if valid
    # This is fast enough (131072 * ~32 checks * 17 bit ops)
    all_candidates = list(range(2 ** N))
    random.shuffle(all_candidates)
    
    added = 0
    for candidate in all_candidates:
        if time.time() > deadline:
            break
        if candidate in words_set:
            continue
        if can_add(candidate, words):
            words.append(candidate)
            words_set.add(candidate)
            added += 1
    
    # If we have time, try local search / improvement
    # Try swapping or modifying existing words to free up space for more
    if time.time() < deadline - 5:
        # Try to find better configurations by removing one word and trying to add multiple
        best_code = list(words)
        best_size = len(words)
        
        # Try removing each word and seeing if we can add more
        for i in range(len(words)):
            if time.time() > deadline:
                break
            
            removed = words[i]
            temp_words = words[:i] + words[i+1:]
            temp_set = words_set - {removed}
            
            # Try to add back other candidates
            new_added = 0
            for candidate in all_candidates:
                if time.time() > deadline:
                    break
                if candidate in temp_set:
                    continue
                if candidate == removed:
                    continue
                if can_add(candidate, temp_words):
                    temp_words.append(candidate)
                    temp_set.add(candidate)
                    new_added += 1
                    if new_added > 1:
                        break  # We improved
            
            if new_added > 1:
                # This removal allowed adding more
                if len(temp_words) > best_size:
                    best_code = list(temp_words)
                    best_size = len(temp_words)
                # Put the removed word back for next iteration
                # (we already have it in words)
        
        if len(best_code) > len(words):
            words = best_code
            words_set = set(words)
    
    return words
# EVOLVE-BLOCK-END
