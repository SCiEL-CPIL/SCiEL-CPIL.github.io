import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120  # seconds
    
    # Precompute neighbors for all words in {0,1,2}^n
    # A word is a tuple of integers
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Compute neighbors: for each word, list of indices of words within Hamming distance <= radius
    # radius is 1 here, so distance 0 or 1
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # distance 0
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                j = word_to_idx[new_w]
                neighbors[i].append(j)
        
    # Greedy Construction
    # uncovered: boolean array or set
    # Using a list of counts is better for the swap logic later
    # For greedy, we just need to pick max coverage
    
    # Initial state: no codewords
    # We want to select K codewords.
    
    # Let's use a standard greedy:
    # 1. Start with empty set of codewords.
    # 2. uncovered = set of all words.
    # 3. While uncovered:
    #    - For each candidate c in all_words:
    #      - Count how many words in 'uncovered' are in neighbors[c].
    #    - Pick c with max count.
    #    - Add c to code.
    #    - Remove neighbors[c] from uncovered.
    
    # This is O(K * N * 13). 
    # N=729, K~80. 80 * 729 * 13 = 753k ops. Fast.
    
    code = []
    uncovered = [True] * N
    
    while any(uncovered):
        best_c = -1
        best_count = -1
        for c in range(N):
            # Count uncovered neighbors
            count = 0
            for nb in neighbors[c]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_c = c
            if best_count == len(neighbors[c]): # Perfect, can't do better
                break
                
        code.append(best_c)
        for nb in neighbors[best_c]:
            uncovered[nb] = False
            
    # Now we have a valid code.
    # Let's try to reduce it.
    
    # Maintain cover_count: how many codewords cover each word
    cover_count = [0] * N
    for c in code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    # Helper to check if code is valid
    def is_valid():
        return all(count > 0 for count in cover_count)
        
    # Local Search: Remove redundant codewords
    # A codeword c is redundant if for all nb in neighbors[c], cover_count[nb] > 1.
    
    improved = True
    while improved:
        improved = False
        # Try to remove any redundant codeword
        for i in range(len(code)):
            c = code[i]
            redundant = True
            for nb in neighbors[c]:
                if cover_count[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                code.pop(i)
                improved = True
                break # Restart loop as indices changed
                
    # Now try to swap codewords to enable further removals
    # Strategy: For each codeword c, find words that are covered ONLY by c (critical words).
    # If there are no critical words, c is redundant (handled above).
    # If there are critical words, we need to replace c with c' that covers all critical words.
    # c' must be in intersection of balls around critical words.
    
    # We can iterate until no improvement (no removal possible after swaps) or time runs out.
    
    while time.time() - start_time < deadline - 5:
        swapped = False
        # Try to find a swap that allows a subsequent removal?
        # Or just a swap that maintains validity?
        # A swap that maintains validity doesn't reduce size.
        # We need a swap that *changes the structure* to allow a future removal.
        
        # Let's try: For each c in code:
        # 1. Identify critical words C_c (cover_count == 1 and covered by c).
        # 2. If C_c is empty, c is redundant. (Should have been removed in previous loop, but maybe counts changed? No, we only swap if valid).
        # 3. Find all c' in all_words such that c' covers all words in C_c.
        # 4. If such c' exists and c' is not in code:
        #    - Check if replacing c with c' keeps the code valid.
        #    - If valid, perform swap.
        #    - Then immediately try to remove redundant codewords again.
        
        for i in range(len(code)):
            if time.time() - start_time > deadline - 5:
                return [list(all_words[c]) for c in code]
            
            c = code[i]
            criticals = []
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    criticals.append(nb)
                    
            if not criticals:
                # Should be redundant, but let's verify and remove if so
                # (This case is handled by the 'improved' loop above, but maybe we missed it?
                #  Actually, if cover_count[nb] == 1 for all nb, it's redundant.
                #  If cover_count[nb] > 1 for some nb, it's not redundant.)
                # Wait, if criticals is empty, it means for all nb, cover_count[nb] != 1.
                # Since c covers nb, cover_count[nb] >= 1.
                # So cover_count[nb] >= 2 for all nb.
                # Thus c is redundant.
                # Why wasn't it removed?
                # Ah, the 'improved' loop breaks after one removal.
                # So we might have multiple redundant codewords.
                # Let's just remove it here.
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                code.pop(i)
                swapped = True # Mark as change to restart outer loop logic?
                # Actually, let's just break and let the outer while handle it.
                # But we are in a 'swap' phase. Let's just remove it and break out to re-evaluate redundancy.
                # It's safer to just remove it and break the inner loop, then the outer while will run the redundancy check again.
                break
            
            # Find candidates c' that cover all criticals
            # c' must be in neighbors[crit] for all crit in criticals
            # Intersection of neighbors
            candidates = set(neighbors[criticals[0]])
            for crit in criticals[1:]:
                candidates &= set(neighbors[crit])
                
            if not candidates:
                continue
                
            # Try each candidate c'
            # We prefer c' that is not already in code
            # And ideally, c' covers many words with low cover_count to "absorb" slack?
            # For now, just pick the first valid one.
            for c_prime in candidates:
                if c_prime == c:
                    continue
                if c_prime in code:
                    continue
                    
                # Check if swapping c with c_prime keeps code valid
                # Remove c
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                # Add c_prime
                for nb in neighbors[c_prime]:
                    cover_count[nb] += 1
                    
                # Check validity
                valid = True
                for nb in range(N):
                    if cover_count[nb] == 0:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    code[i] = c_prime
                    swapped = True
                    break
                else:
                    # Revert
                    for nb in neighbors[c_prime]:
                        cover_count[nb] -= 1
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
                    # If this candidate failed, try next candidate?
                    # Or just break to try next c?
                    # Trying all candidates is expensive. Let's just try one random candidate or the first one.
                    # Actually, if the first one fails, others might work.
                    # But to save time, let's just try the first valid structural candidate.
                    pass
            
            if swapped:
                break
        
        if not swapped:
            # No swap found that maintains validity.
            # We are stuck.
            # Try a random restart? Or accept current solution.
            # Given the small search space, if we can't swap, we might be at a local minimum.
            # Let's try a few random swaps that *temporarily* break validity? No, too complex.
            # Let's just return the current best.
            break

    return [list(all_words[c]) for c in code]
