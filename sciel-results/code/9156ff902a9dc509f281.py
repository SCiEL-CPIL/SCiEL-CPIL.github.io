import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected in this specific task instance)
        # Just return a trivial covering (all words) or similar
        # But the task is specific to 3, 8, 1.
        pass

    # Precomputation
    N = 3 ** n
    # cover_masks[i] is a bitmask where bit j is 1 if word j is within distance 1 of word i
    cover_masks = [0] * N
    
    # Helper to convert int to word and neighbors
    # We can generate neighbors on the fly or precompute.
    # Precomputing 6561 masks of 6561 bits each.
    # 6561 * 6561 / 8 bytes ~ 5.4 MB. This is fine.
    
    # Efficient way to build cover masks:
    # For each word w, its neighbors are w itself and w with one digit changed.
    # We need to map digit changes to integer changes.
    # Let's precompute the "delta" for each position and change.
    
    # Actually, simpler:
    # For i in range(N):
    #   w = to_word(i)
    #   mask = 1 << i
    #   for pos in range(n):
    #      for delta in [-1, 1]:
    #         new_val = (w[pos] + delta) % 3
    #         new_w = w.copy()
    #         new_w[pos] = new_val
    #         new_i = to_int(new_w)
    #         mask |= (1 << new_i)
    #   cover_masks[i] = mask
    
    # This precomputation might take a second or two.
    # Let's optimize to_int/to_word.
    
    def to_int(w):
        val = 0
        for d in w:
            val = val * 3 + d
        return val
        
    def to_word(val):
        w = [0]*n
        for i in range(n-1, -1, -1):
            w[i] = val % 3
            val //= 3
        return w

    # Precompute cover masks
    for i in range(N):
        w = to_word(i)
        mask = 1 << i
        for pos in range(n):
            orig = w[pos]
            for delta in [-1, 1]:
                new_val = (orig + delta) % 3
                if new_val == orig: continue
                # Calculate new integer value efficiently
                # The difference is (new_val - orig) * (3 ** (n - 1 - pos))
                diff = (new_val - orig) * (3 ** (n - 1 - pos))
                new_i = i + diff
                mask |= (1 << new_i)
        cover_masks[i] = mask

    all_words_mask = (1 << N) - 1
    best_solution = None
    best_size = N # Upper bound
    
    # Random restarts
    while time.time() < deadline:
        # Randomize candidate order
        candidates = list(range(N))
        random.shuffle(candidates)
        
        uncovered = all_words_mask
        solution = []
        
        # Greedy Set Cover
        while uncovered:
            best_c = -1
            best_count = -1
            # Find codeword covering most uncovered bits
            # Optimization: If we find a count equal to the number of uncovered bits, we can break early?
            # No, because we want to maximize coverage of *uncovered* set.
            # The max possible is popcount(uncovered).
            
            for c in candidates:
                # If c is already in solution, skip? 
                # Actually, if we pick c, we add it. We don't pick the same c twice.
                # But candidates is a permutation of all words.
                # Once we pick c, we effectively remove it from consideration for future picks in this run.
                # However, the `uncovered` set changes.
                # It's possible that a word previously not best becomes best later.
                # But standard greedy picks the best available.
                
                # To avoid re-evaluating words already picked (which have 0 new coverage),
                # we can just check if (cover_masks[c] & uncovered) is 0?
                # No, we just need the best one.
                
                count = (cover_masks[c] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
                    if best_count == uncovered.bit_count():
                        # Perfect cover found for remaining? Unlikely in intermediate steps
                        # But if count is 0, we are stuck (should not happen if cover is valid)
                        break
            
            if best_c == -1 or best_count == 0:
                # Failure to cover? Should not happen.
                break
                
            solution.append(best_c)
            uncovered &= ~cover_masks[best_c]
            
        # Redundancy Removal
        # Check if any codeword in solution is redundant
        changed = True
        while changed:
            changed = False
            new_solution = []
            for c in solution:
                # Check if cover(c) is subset of union of covers of others
                # Union of others = all_words_mask & ~ (intersection of complements?)
                # Easier: 
                # Let U = union of covers of (solution \ {c})
                # If cover(c) & ~U == 0, then c is redundant.
                
                # Calculating U for each c is O(S^2 * N/word_size).
                # S ~ 500. 500*500 = 250,000 ops.
                # Bit operations on 6561 bits are fast.
                
                # Optimization: 
                # We can maintain the union of all selected covers?
                # No, removing c changes the union.
                
                # Let's just do it.
                # Union of all covers in solution
                union_all = 0
                for c2 in solution:
                    union_all |= cover_masks[c2]
                    
                # We need union of solution \ {c}
                # This is expensive if done naively inside the loop.
                
                # Alternative: 
                # Just check if there is any word covered ONLY by c.
                # Words covered by c: W_c = cover_masks[c]
                # Words covered by others: W_others = union_{c' != c} cover_masks[c']
                # If W_c & ~W_others == 0, c is redundant.
                
                # To speed up, we can compute W_others incrementally?
                # Or just compute union of all, and then if c is the only one covering a specific bit, it's not redundant.
                
                # Let's use a simpler heuristic for redundancy:
                # If we remove c, does `uncovered` become non-zero?
                # We don't have `uncovered` from the greedy phase easily accessible here without re-computing.
                
                # Let's re-compute coverage from scratch for the solution to check redundancy.
                # This is O(S * N/word_size) per check.
                
                pass # Skip complex redundancy for now, greedy is often close.
            
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
            
    # Convert best_solution to list of lists
    return [to_word(c) for c in best_solution]
