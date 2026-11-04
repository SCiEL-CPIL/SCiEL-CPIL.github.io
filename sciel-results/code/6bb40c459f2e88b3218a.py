"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    For the canonical task q=3, n=6, radius=1, we aim for ~73 words.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # 1. Generate the perfect ternary tetracode (length 4, radius 1, size 9)
    # This is a known perfect code.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # 2. Initial Solution: Direct product with all tails (length n-4)
    # Size: 9 * 3^(n-4)
    if n <= 4:
        # Fallback for small n (though task is n=6)
        # Simple greedy or exhaustive if small
        all_words = list(itertools.product(range(q), repeat=n))
        code = []
        uncovered = set(all_words)
        for w in all_words:
            if not uncovered:
                break
            if w in uncovered:
                code.append(list(w))
                # Remove neighbors
                # For efficiency in small n, just check
                for u in list(uncovered):
                    if sum(1 for i in range(n) if u[i] != w[i]) <= radius:
                        uncovered.remove(u)
        return code

    base_tails = list(itertools.product(range(3), repeat=n - 4))
    
    # Initial code: tetracode head + all tails
    # This is a valid covering code because tetracode covers all heads, and tails are exact matches.
    # Wait, direct product: (h, t) is in code.
    # A word (w_head, w_tail) is covered if exists (h, t) in code such that d(w_head, h) + d(w_tail, t) <= radius.
    # If t = w_tail, then d(w_tail, t) = 0. Then we need d(w_head, h) <= radius.
    # Since tetracode is a perfect code of radius 1, every w_head is within distance 1 of some h.
    # So yes, this is a valid covering code.
    initial_code = [h + list(t) for h in tetracode for t in base_tails]
    
    # 3. Greedy Removal (Local Search)
    # We want to remove words from the code while maintaining the covering property.
    # A word can be removed if every word in the space is still covered by the remaining code.
    
    # To speed up coverage checking, we can use a bitset or a set of covered words.
    # Space size is 3^6 = 729. This is small.
    # We can maintain a count of how many code words cover each space word.
    # If count[w] > 1 for all w, we can remove any code word.
    # Actually, we need count[w] >= 1 for all w.
    # If we remove c, we decrement count[w] for all w in Ball(c, radius).
    # If any count[w] becomes 0, removal is invalid.
    
    # Let's represent the space as integers 0 to 3^n - 1.
    # Mapping word -> int: sum(w[i] * 3^(n-1-i))
    # Mapping int -> word: standard base-3 expansion.
    
    # Precompute balls for the initial code? Or compute on the fly?
    # n=6, radius=1. Ball size is 1 + 6*(3-1) = 13 words.
    # This is very small.
    
    def word_to_int(w):
        val = 0
        for x in w:
            val = val * 3 + x
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % 3
            val //= 3
        return w

    # Precompute neighbors for all words in the space?
    # There are 729 words.
    # For each word, compute its ball (itself + 1-dist neighbors).
    # Store as list of ints.
    
    all_ints = range(3 ** n)
    neighbors_cache = {}
    
    # Efficiently generate neighbors
    # For a word w, neighbors are w itself, and w with one digit changed.
    # Since n is small, we can just iterate.
    
    # Let's precompute the "coverage count" array.
    # counts[i] = number of code words covering word i.
    
    code_set = set(map(tuple, initial_code))
    counts = np.zeros(3 ** n, dtype=np.int32)
    
    # Build initial counts
    # For each code word, add 1 to all its neighbors.
    # Optimization: Precompute the neighbor lists for all 729 words.
    # This takes 729 * 13 operations, which is trivial.
    
    neighbor_lists = []
    for i in all_ints:
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    w[pos] = new_val
                    ball.append(word_to_int(w))
            w[pos] = orig
        neighbor_lists.append(ball)
    
    # Initialize counts
    for c in initial_code:
        ci = word_to_int(c)
        for nb in neighbor_lists[ci]:
            counts[nb] += 1
            
    # Greedy Removal Loop
    # We try to remove words.
    # Strategy: Iterate through code words. If a word can be removed, remove it.
    # To be more effective, we can try to remove words that are "redundant" the most.
    # A word is redundant if for all w in Ball(c), counts[w] > 1.
    # If we remove c, counts[w] decreases by 1. We need counts[w] - 1 >= 1 => counts[w] >= 2.
    
    # We'll use a list for the code to allow removal.
    # However, removing from a list is O(N). N is ~162 initially. This is fine.
    # But we need to update counts.
    
    # Let's maintain a list of active code words.
    active_code = list(initial_code)
    active_indices = set(range(len(active_code)))
    
    # To make removal efficient, we can just swap with last and pop.
    # But we need to map code word to its index in the list for quick access?
    # Actually, we can just iterate.
    
    # Better: Keep a list of code words.
    # When we remove one, we update counts.
    
    # Let's try a simple greedy:
    # While possible:
    #   Find a word c in code such that min(counts[w] for w in Ball(c)) >= 2.
    #   Remove c.
    
    # To find such a c quickly:
    # We can check each c in the current code.
    
    # Optimization:
    # If a word w has counts[w] == 1, it is "critical".
    # Any code word c that covers w cannot be removed.
    # So we only consider c such that for all w in Ball(c), counts[w] >= 2.
    
    # Let's implement this.
    
    removed = 0
    while time.time() < deadline - 5.0:  # Stop 5s early
        found_removal = False
        # Iterate through a copy of the code to avoid modification issues during iteration
        # Or iterate in reverse?
        # Since we might remove one, then the next index shifts.
        # Let's just pick a candidate, remove it, and restart the scan or continue carefully.
        # Given small N, restarting or scanning a list is fine.
        
        # Sort candidates by "redundancy"?
        # Words with higher min-count in their ball are safer to remove?
        # Or just pick the first one found.
        
        # Let's try to pick the one with the highest "slack" to be more robust?
        # Or just the first one.
        
        for idx in range(len(active_code)):
            c = active_code[idx]
            ci = word_to_int(c)
            ball = neighbor_lists[ci]
            
            # Check if all neighbors have count >= 2
            can_remove = True
            for nb in ball:
                if counts[nb] < 2:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                # Update counts
                for nb in ball:
                    counts[nb] -= 1
                
                # Remove from list
                active_code.pop(idx)
                removed += 1
                found_removal = True
                break # Restart scan from beginning as indices shifted
        
        if not found_removal:
            # No more simple removals possible
            # Try a more sophisticated move?
            # E.g. swap? Or just stop.
            # For now, stop.
            break
            
    return active_code
# EVOLVE-BLOCK-END
