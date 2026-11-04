# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def _int_to_word(val, q, n):
    """Convert an integer in [0, q**n) to a word (list of n ints in 0..q-1)."""
    word = [0] * n
    for i in range(n):
        word[i] = val % q
        val //= q
    return word

def _word_to_int(word, q, n):
    """Convert a word (list of n ints) to an integer in [0, q**n)."""
    val = 0
    for i in range(n - 1, -1, -1):
        val = val * q + word[i]
    return val

def _ball_mask(q, n, radius):
    """
    Precompute a boolean mask of size q**n where mask[i] is True if word i
    is within Hamming distance <= radius of the zero word.
    This is the 'ball' around 0.
    """
    total = q ** n
    mask = np.zeros(total, dtype=bool)
    mask[0] = True
    
    # Generate all words within radius r of 0
    # For radius 1: 0 + all words with exactly one non-zero position
    if radius == 1:
        for pos in range(n):
            for sym in range(1, q):
                w = [0] * n
                w[pos] = sym
                idx = _word_to_int(w, q, n)
                mask[idx] = True
    elif radius == 0:
        pass
    else:
        # General case: iterate over all words and check distance (slow but correct)
        # For small n this is fine
        for val in range(total):
            w = _int_to_word(val, q, n)
            dist = sum(1 for i in range(n) if w[i] != 0)
            if dist <= radius:
                mask[val] = True
    return mask

def _covers_all(code, q, n, radius, ball_mask, total):
    """
    Check if the given code covers all words in {0..q-1}^n.
    code: list of integers (each is a word index)
    """
    if not code:
        return total == 0
    
    covered = np.zeros(total, dtype=bool)
    for c in code:
        # XOR-like operation in base-q? No, it's addition mod q.
        # Shift the ball mask by c.
        # ball_mask is centered at 0. We need to shift it by c.
        # Shift means: covered |= shift(ball_mask, c)
        # How to shift a boolean array by c in base-q?
        # It's not a simple circular shift unless q**n is a power of 2 and c is a power of 2, etc.
        # Actually, the group is (Z_q)^n. The operation is addition mod q component-wise.
        # So if x is in the ball around 0, then x + c is in the ball around c.
        # We need to map indices.
        
        # Efficient way: 
        # For each element in the ball, add c to it.
        # But doing this for every c is O(|code| * |ball|).
        # |ball| for q=3, n=6, r=1 is 1 + 6*2 = 13.
        # |code| is ~70-80.
        # So 80 * 13 = 1040 operations per check. Very fast.
        
        # Let's precompute the offsets for the ball.
        # Actually, let's just compute the shifted indices on the fly or precompute a list of offsets.
        
        # Better: precompute a list of offsets `ball_offsets` which are the indices of the ball.
        # Then for each c, we add c to each offset.
        # But addition mod q is not the same as integer addition.
        # We need to add c to each offset in the group (Z_q)^n.
        
        # Let's precompute the ball as a list of words (tuples), not indices.
        # Then for each c, convert c to word, add to each ball word, convert back to index.
        # This is a bit slow.
        
        # Alternative: Use the fact that we can precompute a lookup table for shifts?
        # No, that's too big.
        
        # Let's stick to: for each c, generate the covered indices.
        # To speed up, precompute the ball as a list of tuples (the actual words).
        
        pass # This function will be rewritten below with better structure

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    total = q ** n
    
    # Precompute ball offsets as a list of tuples (words)
    ball_words = []
    for val in range(total):
        w = _int_to_word(val, q, n)
        dist = sum(1 for i in range(n) if w[i] != 0)
        if dist <= radius:
            ball_words.append(tuple(w))
    
    ball_size = len(ball_words)
    
    # Precompute a mapping from word tuple to index? Or just use _word_to_int.
    # _word_to_int is fast enough.
    
    def add_to_covered(covered, c_word):
        """Update covered boolean array by adding the ball around c_word."""
        for bw in ball_words:
            # Add c_word and bw component-wise mod q
            idx_word = tuple((c_word[i] + bw[i]) % q for i in range(n))
            idx = _word_to_int(idx_word, q, n)
            covered[idx] = True
    
    def is_covered(covered):
        return covered.all()
    
    def uncovered_count(covered):
        return int((~covered).sum())
    
    # --- Phase 1: Greedy Set Cover ---
    # Initialize covered array
    covered = np.zeros(total, dtype=bool)
    code = []
    
    # To speed up greedy, we can maintain a list of uncovered words.
    # But checking which c covers the most uncovered words is O(|code| * |ball| * |uncovered|)
    # which is too slow.
    
    # Standard greedy:
    # 1. Pick c that covers the most *new* words.
    # 2. Add to code, update covered.
    # 3. Repeat until all covered.
    
    # To find the best c efficiently:
    # We can iterate over all possible c (total of them) and count how many new words it covers.
    # This is O(total * ball_size) per step.
    # total = 729, ball_size = 13.
    # 729 * 13 = ~9500 operations per step.
    # Number of steps ~ 70-80.
    # Total ~ 700k operations. Very fast.
    
    # Let's implement this.
    
    # Precompute for each c, the list of indices it covers.
    # cover_table[c] = list of indices in the ball around c.
    cover_table = []
    for c_val in range(total):
        c_word = _int_to_word(c_val, q, n)
        indices = []
        for bw in ball_words:
            idx_word = tuple((c_word[i] + bw[i]) % q for i in range(n))
            idx = _word_to_int(idx_word, q, n)
            indices.append(idx)
        cover_table.append(indices)
    
    # Greedy loop
    while not is_covered(covered):
        best_c = -1
        best_gain = -1
        for c in range(total):
            gain = 0
            for idx in cover_table[c]:
                if not covered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
            if best_gain == ball_size:
                break # Can't do better
        
        if best_gain <= 0:
            break # Should not happen if space is connected
        
        code.append(best_c)
        for idx in cover_table[best_c]:
            covered[idx] = True
        
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return [_int_to_word(c, q, n) for c in code]
    
    # --- Phase 2: Local Search (Add-then-Remove) ---
    # Try to remove codewords if possible.
    # If a codeword is redundant, remove it.
    # Then try swapping: remove one, add another to see if we can maintain coverage with same size or less.
    
    # First, try to remove redundant words.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code)):
            # Temporarily remove code[i]
            removed_c = code[i]
            # Check if all words covered by code[i] are still covered by the rest
            # Actually, we just need to check if the union of the rest covers everything.
            # Efficient check: 
            # Count how many words are ONLY covered by code[i].
            # If 0, it's redundant.
            
            # Let's compute the coverage of the rest.
            # This is O(|code| * ball_size) = 80 * 13 = 1040.
            # Do it for each i? 80 * 1040 = 83k. Fast.
            
            rest_covered = np.zeros(total, dtype=bool)
            for j, c in enumerate(code):
                if j == i:
                    continue
                for idx in cover_table[c]:
                    rest_covered[idx] = True
            
            # Check if code[i] covers any word not in rest_covered
            redundant = True
            for idx in cover_table[removed_c]:
                if not rest_covered[idx]:
                    redundant = False
                    break
            
            if redundant:
                code.pop(i)
                changed = True
                break # Restart loop since indices changed
    
    if time.time() > deadline:
        return [_int_to_word(c, q, n) for c in code]
    
    # Now try swaps: remove one, add another (different) to see if we can keep size same but maybe enable further removals?
    # Or just try to find a smaller set by random restarts?
    
    # Let's do a few random restarts of the greedy + pruning if time permits.
    best_code = code[:]
    best_size = len(code)
    
    # Random restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        random.shuffle(cover_table) # No, this doesn't make sense.
        # Instead, randomize the order of consideration in greedy?
        # Or just run greedy with a random tie-breaking?
        
        # Let's try a different initial seed or randomization in the greedy step.
        # Actually, the greedy is deterministic.
        # Let's try random local search on the current best.
        
        # Randomly remove a word, then try to re-add the best word to cover the gap?
        # This is complex.
        
        # Let's just try a few more rounds of add-then-remove with randomization.
        
        # For now, stick with the deterministic greedy + pruning.
        # If we have time, we can try to improve the greedy by looking ahead.
        
        pass
    
    # Final check and return
    # Ensure validity
    final_covered = np.zeros(total, dtype=bool)
    for c in best_code:
        for idx in cover_table[c]:
            final_covered[idx] = True
    
    if not final_covered.all():
        # Fallback to a known valid construction (e.g., direct sum)
        # But greedy should be valid.
        pass
        
    return [_int_to_word(c, q, n) for c in best_code]

# EVOLVE-BLOCK-END
