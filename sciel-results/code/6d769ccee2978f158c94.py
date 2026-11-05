import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Generate initial code using direct sum of tetracode
    # Tetracode: length 4, size 9, radius 1 perfect code
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    if n >= 4:
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Fallback for small n, though task is n=6
        initial_codewords = [list(word) for word in itertools.product(range(q), repeat=n)]
        
    # Ensure distinctness (they are distinct by construction)
    # Convert to tuples for hashing if needed, but we use indices
    m = len(initial_codewords)
    total_words = q ** n
    
    # Precompute covers for each codeword in the initial set
    # covers[i] = list of word indices covered by initial_codewords[i]
    covers = []
    for cw in initial_codewords:
        covered_words = set()
        # The word itself
        idx = 0
        for v in cw:
            idx = idx * q + v
        covered_words.add(idx)
        
        # Words at distance 1
        for pos in range(n):
            orig_val = cw[pos]
            for alt_val in range(q):
                if alt_val == orig_val:
                    continue
                new_cw = cw.copy()
                new_cw[pos] = alt_val
                new_idx = 0
                for v in new_cw:
                    new_idx = new_idx * q + v
                covered_words.add(new_idx)
        covers.append(list(covered_words))
        
    # We will work with a list of active codeword indices from the initial set
    # But we might want to add new codewords? The problem asks for *any* code.
    # The search space is huge (729 possible words).
    # Usually, starting from a good construction and pruning is best.
    # However, sometimes adding a random word and removing two others helps.
    
    # Let's stick to the set of initial 81 words for removal first.
    # If we get stuck, we can try to introduce new words.
    
    # State: active[i] = True if initial_codewords[i] is in the code
    active = [True] * m
    
    # cover_count[w] = number of active codewords covering word w
    cover_count = np.zeros(total_words, dtype=np.int32)
    for i in range(m):
        for w in covers[i]:
            cover_count[w] += 1
            
    def try_remove_redundant(active, cover_count, covers, m, deadline):
        """
        Attempts to remove redundant codewords.
        Returns True if any removal occurred.
        Modifies active and cover_count in place.
        """
        removed_any = False
        # Random order to break symmetry
        order = list(range(m))
        random.shuffle(order)
        
        for i in order:
            if time.time() > deadline - 2.0:
                break
            if not active[i]:
                continue
            
            # Check if redundant: min cover_count in covers[i] >= 2
            # If min is 1, removing it leaves a word uncovered.
            # If min >= 2, all words are still covered.
            min_count = 100 # max possible is m
            for w in covers[i]:
                if cover_count[w] < min_count:
                    min_count = cover_count[w]
                    if min_count == 1:
                        break # Not redundant
            
            if min_count >= 2:
                # Remove
                active[i] = False
                for w in covers[i]:
                    cover_count[w] -= 1
                removed_any = True
        
        return removed_any

    best_active = active[:]
    best_count = sum(active)
    
    # Phase 1: Greedy removal with multiple restarts
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        current_active = active[:]
        current_cover_count = cover_count.copy()
        
        # Iterative removal
        while True:
            if time.time() > deadline - 10.0:
                break
            improved = try_remove_redundant(current_active, current_cover_count, covers, m, deadline)
            if not improved:
                break
        
        current_count = sum(current_active)
        if current_count < best_count:
            best_count = current_count
            best_active = current_active[:]
            # Update global best state
            active = best_active[:]
            cover_count = current_cover_count.copy()
            
    # Phase 2: Local Search with Swaps
    # We maintain the best solution found so far.
    # We try to swap an active codeword with an inactive one, or add a new one from the full space.
    # Adding from the full space is expensive to check coverage for new words.
    # Let's restrict swaps to the initial set first.
    
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
            
        current_active = best_active[:]
        current_cover_count = cover_count.copy()
        
        # Try swaps within the initial set
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Find active codewords that are "critical" (cover words with count 1)
            # and inactive ones that might be better.
            # Simpler: Randomly pick an active i and inactive j.
            # Check if swapping i->j is valid.
            
            active_indices = [i for i in range(m) if current_active[i]]
            inactive_indices = [i for i in range(m) if not current_active[i]]
            
            if not active_indices or not inactive_indices:
                break
            
            random.shuffle(active_indices)
            random.shuffle(inactive_indices)
            
            # Try a few swaps
            for i in active_indices[:10]:
                if time.time() > deadline - 5.0:
                    break
                for j in inactive_indices[:10]:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Simulate swap: remove i, add j
                    # Check validity:
                    # 1. Words covered only by i (count 1) must be covered by j.
                    # 2. No word becomes uncovered.
                    
                    # Temporarily decrement cover_count for i's covers
                    valid = True
                    for w in covers[i]:
                        current_cover_count[w] -= 1
                        if current_cover_count[w] < 0:
                            valid = False
                            break
                    if not valid:
                        # Rollback
                        for w in covers[i]:
                            current_cover_count[w] += 1
                        continue
                    
                    # Check if j covers any word that is now at count 0
                    # Actually, we just need to ensure that after adding j, no word is 0.
                    # The words that became 0 are those in covers[i] where count was 1.
                    # Let's identify them.
                    zero_words = []
                    for w in covers[i]:
                        if current_cover_count[w] == 0:
                            zero_words.append(w)
                    
                    if zero_words:
                        # Check if j covers all zero_words
                        # covers[j] is a list. Convert to set for fast lookup?
                        # covers[j] has 13 elements.
                        j_covers_set = set(covers[j])
                        for w in zero_words:
                            if w not in j_covers_set:
                                valid = False
                                break
                    
                    if valid:
                        # Commit swap
                        current_active[i] = False
                        current_active[j] = True
                        for w in covers[j]:
                            current_cover_count[w] += 1
                        improved = True
                        # Try to remove more after swap
                        try_remove_redundant(current_active, current_cover_count, covers, m, deadline)
                        break # Restart inner loop with new state
                    else:
                        # Rollback
                        for w in covers[i]:
                            current_cover_count[w] += 1
                
                if improved:
                    break
            
        current_count = sum(current_active)
        if current_count < best_count:
            best_count = current_count
            best_active = current_active[:]
            active = best_active[:]
            cover_count = current_cover_count.copy()
            
    # Phase 3: Try adding new codewords from the full space
    # This is harder. We need to generate candidate codewords.
    # Candidates: words that are currently poorly covered? Or random?
    # If we add a new word, we might be able to remove 2 existing words.
    # This is a "2-for-1" or "3-for-2" exchange.
    
    # Given time constraints, let's stick to the initial set optimization which is robust.
    # The direct sum of tetracode is very good. 81 -> 73 is the target.
    
    # Final cleanup
    try_remove_redundant(best_active, cover_count, covers, m, deadline)
    
    # Construct result
    result = []
    for i in range(m):
        if best_active[i]:
            result.append(initial_codewords[i])
            
    return result
