import time
import random
import itertools

def construct(q, n, radius):
    # Task specific: q=3, n=6, radius=1
    # But we should handle general inputs if possible, though the task is specific.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So q, n, radius are fixed for this run, but we should be robust.
    
    total_words = q ** n
    if total_words > 10000:
        # Fallback to simple construction if too large for exact search
        # For this task, 3^6 = 729, so we are fine.
        pass
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120 # 120s limit, keep margin
    
    # Precompute covers
    # Word is a tuple of n ints, or an integer in base q.
    # Let's use integer representation for speed.
    
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n):
            w[n-1-i] = val % q
            val //= q
        return w

    # Generate all words as ints
    all_words = list(range(total_words))
    
    # Precompute neighbors (Hamming distance <= radius)
    # For radius 1, neighbors are self + change one position to other q-1 values.
    # masks[w] is a bitmask of words covered by w
    masks = [0] * total_words
    
    # To compute masks efficiently:
    # For each word w, generate all words within distance 1.
    for w_int in all_words:
        w_tuple = tuple(int_to_word(w_int))
        mask_val = 0
        # Self
        mask_val |= (1 << w_int)
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w_tuple[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_tuple_list = list(w_tuple)
                    new_tuple_list[pos] = new_val
                    new_tuple = tuple(new_tuple_list)
                    new_int = word_to_int(new_tuple)
                    mask_val |= (1 << new_int)
                    
        if radius >= 2:
            # Not needed for this task, but good for robustness
            # This part is expensive, so we'll only do radius 1 specifically or skip if radius > 1 and n is large
            # For this task radius is 1.
            pass
            
        masks[w_int] = mask_val

    all_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = float('inf')
    
    # Randomized Greedy Set Cover
    # We run multiple restarts.
    # Each restart:
    # 1. uncovered = all_mask
    # 2. code = []
    # 3. while uncovered:
    #      find w maximizing popcount(masks[w] & uncovered)
    #      tie-break randomly
    #      add w to code
    #      uncovered &= ~masks[w]
    
    import sys
    # Use a helper to count bits
    # Python 3.8+ has int.bit_count()
    
    num_restarts = 100 # We can do more if time permits
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        uncovered = all_mask
        code = []
        
        while uncovered:
            # Find best w
            max_count = -1
            candidates = []
            
            # Optimization: only consider w that cover at least one uncovered word
            # We can iterate all w, it's only 729.
            for w in range(total_words):
                # If w is already in code, skip? No, greedy doesn't repeat, but mask is static.
                # However, if a word is covered, it's removed from 'uncovered'.
                # We just need the count of NEW words covered.
                count = (masks[w] & uncovered).bit_count()
                if count > max_count:
                    max_count = count
                    candidates = [w]
                elif count == max_count and count > 0:
                    candidates.append(w)
            
            if not candidates:
                # Should not happen if uncovered is not 0 and masks are correct
                break
                
            # Random tie-break
            w = random.choice(candidates)
            code.append(w)
            uncovered &= ~masks[w]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we found the theoretical lower bound or a very good solution, we might stop early?
            # Lower bound is 71. Best known 73.
            if best_size <= 73:
                # We can continue to try for 72 or 71, but 73 is the target for "improvement" over 81.
                # Let's keep running a bit to see if we can get 72.
                pass

    if best_code is None:
        # Fallback: return the seed solution
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert best_code ints back to lists
    return [int_to_word(w) for w in best_code]
