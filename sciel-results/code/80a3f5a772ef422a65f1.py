"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code for {0,1,2}^n with covering radius 1.
    Uses a bit-packed integer representation and a critical-word swap heuristic.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n <= 4:
            if n == 1:
                return [[i] for i in range(q)]
            elif n == 2:
                return [list(w) for w in itertools.product(range(q), repeat=2)]
            elif n == 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return tetracode
            else:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # General fallback for larger n or different q
        # Use direct sum of tetracodes if possible
        if n % 4 == 0:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            codes = []
            for block in range(n // 4):
                pass # Complex to generalize direct sum easily without recursion
            # Just return a large random set or full space if small
            if q ** n < 10000:
                return [list(w) for w in itertools.product(range(q), repeat=n)]
            else:
                # Greedy random
                all_words = list(itertools.product(range(q), repeat=n))
                random.shuffle(all_words)
                covered = set()
                result = []
                for w in all_words:
                    if w not in covered:
                        result.append(list(w))
                        # Add neighbors to covered
                        for i in range(n):
                            for v in range(q):
                                if v != w[i]:
                                    nw = list(w)
                                    nw[i] = v
                                    covered.add(tuple(nw))
                        covered.add(tuple(w))
                return result

    # Specific optimization for q=3, n=8, radius=1
    # Map words to integers 0..6560
    # word_to_int: tuple -> int
    # int_to_word: int -> tuple
    
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_int = {w: i for i, w in enumerate(all_words)}
    int_to_word = {i: w for i, w in enumerate(all_words)}
    total_words = q ** n # 6561

    # Precompute covers[i]: list of ints covered by codeword i
    covers = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        covers[i].append(i)
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    covers[i].append(word_to_int[tuple(nw)])
    
    # Initial construction: 486 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            cw = h + list(tail)
            initial_codewords.append(word_to_int[tuple(cw)])
    
    current_codewords = set(initial_codewords)
    cov_count = [0] * total_words
    
    # Initialize coverage counts
    for c in current_codewords:
        for w in covers[c]:
            cov_count[w] += 1
            
    def is_removable(c):
        for w in covers[c]:
            if cov_count[w] <= 1:
                return False
        return True

    def try_remove():
        """Try to find and remove a redundant codeword. Returns True if removed."""
        # Shuffle to avoid bias
        cw_list = list(current_codewords)
        random.shuffle(cw_list)
        for c in cw_list:
            if is_removable(c):
                for w in covers[c]:
                    cov_count[w] -= 1
                current_codewords.remove(c)
                return True
        return False

    # Main search loop
    # We try to remove codewords. If stuck, we perform swaps.
    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave buffer for 120s limit
    
    while time.time() - start_time < time_limit:
        # Phase 1: Remove redundant codewords
        removed = False
        while try_remove():
            removed = True
            # Check time
            if time.time() - start_time > time_limit:
                break
        
        if time.time() - start_time > time_limit:
            break

        if not removed and len(current_codewords) < 486:
            # We improved, but might be stuck. Try swaps.
            pass
            
        # Phase 2: If no removal, try a swap
        # Pick a random codeword
        if not current_codewords:
            break
            
        c = random.choice(list(current_codewords))
        
        # Find critical words for c
        crits = []
        for w in covers[c]:
            if cov_count[w] == 1:
                crits.append(w)
                
        if not crits:
            # Should not happen if try_remove failed, but just in case
            continue
            
        # Pick a critical word
        w_crit = random.choice(crits)
        
        # Candidates for replacement: codewords that cover w_crit
        candidates = covers[w_crit]
        random.shuffle(candidates)
        
        swap_performed = False
        for c_new in candidates:
            if c_new == c:
                continue
            
            # Simulate swap: remove c, add c_new
            # Check if this leads to a removable state
            # We need to check if any codeword becomes removable after the swap.
            # The swap affects cov_count for words in covers[c] (decrease) and covers[c_new] (increase).
            
            # Optimization: Only check removability for codewords that share words with covers[c_new] 
            # (since counts increase there) or covers[c] (counts decrease, so less likely to become removable).
            # Actually, if we remove c, counts in covers[c] drop. If they drop to 1, it's bad.
            # If we add c_new, counts in covers[c_new] rise.
            
            # Simple check: 
            # 1. Ensure w_crit is still covered. It is, by c_new.
            # 2. Check if any word in covers[c] becomes uncovered (count 0) if we remove c?
            #    No, we are swapping. So we remove c first? No, simultaneous.
            #    New count for w:
            #    if w in covers[c] and not in covers[c_new]: count - 1
            #    if w in covers[c_new] and not in covers[c]: count + 1
            #    if w in both: count unchanged
            #    if w in neither: count unchanged
            
            # A word w becomes critical (count 1) if it was count 2 and we remove c (and c_new doesn't cover it).
            # A word w becomes uncovered (count 0) if it was count 1 and we remove c (and c_new doesn't cover it).
            # We must ensure no word becomes uncovered.
            
            valid_swap = True
            # Check for uncovering
            for w in covers[c]:
                if w not in covers[c_new]: # Approximation: check intersection?
                    # We need to know if c_new covers w
                    # covers is a list, so 'w in covers[c_new]' is O(N). 
                    # Better: use a set for covers? Or precomputed bitsets?
                    # With 6561 words, 'in' on a list of ~29 items is fast.
                    if cov_count[w] == 1:
                        # If c_new doesn't cover w, and count is 1, removing c makes it 0.
                        # Check if c_new covers w
                        # Since we are iterating candidates from covers[w_crit], and w_crit is in covers[c],
                        # c_new covers w_crit.
                        # But w might be different.
                        if w_crit != w:
                            # Check if c_new covers w
                            # This check is expensive if done naively for every w in covers[c]
                            # But covers[c] is small (29).
                            # Is w in covers[c_new]?
                            if w not in covers[c_new]:
                                valid_swap = False
                                break
            
            if not valid_swap:
                continue
                
            # If valid, check if it enables a removal
            # Apply swap temporarily
            for w in covers[c]:
                cov_count[w] -= 1
            for w in covers[c_new]:
                cov_count[w] += 1
                
            current_codewords.remove(c)
            current_codewords.add(c_new)
            
            # Try to remove
            if try_remove():
                swap_performed = True
                break
            else:
                # Revert swap
                current_codewords.remove(c_new)
                current_codewords.add(c)
                for w in covers[c]:
                    cov_count[w] += 1
                for w in covers[c_new]:
                    cov_count[w] -= 1
                    
        if not swap_performed:
            # If we tried all candidates and none allowed a removal, 
            # maybe we should accept a swap that doesn't immediately allow removal 
            # but improves the "critical count" or is random to escape local minima?
            # For now, let's just break or try a random swap that maintains validity.
            # To avoid infinite loop, let's just break if we can't improve.
            # Or, try one random valid swap that minimizes the increase in critical words.
            best_c_new = None
            best_crit_increase = float('inf')
            
            for c_new in candidates:
                if c_new == c:
                    continue
                # Check validity (no uncovering)
                valid = True
                for w in covers[c]:
                    if cov_count[w] == 1 and w not in covers[c_new]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate change in critical words
                # This is expensive. 
                # Let's just pick a random valid one to escape local minima occasionally.
                if random.random() < 0.05:
                    # Apply swap
                    for w in covers[c]:
                        cov_count[w] -= 1
                    for w in covers[c_new]:
                        cov_count[w] += 1
                    current_codewords.remove(c)
                    current_codewords.add(c_new)
                    swap_performed = True
                    break

            if not swap_performed:
                # Stuck. Maybe restart with a different random seed or break?
                # Breaking might be premature. 
                # Let's try a different codeword c.
                pass # The while loop continues, picking a new random c.

    # Convert back to lists
    return [list(int_to_word[c]) for c in current_codewords]

# EVOLVE-BLOCK-END
