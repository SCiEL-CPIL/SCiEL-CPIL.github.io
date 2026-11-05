import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Total words in space
    total_words = q ** n
    
    # Precompute neighbors for each word
    # word is an integer 0..total_words-1
    # neighbors[word] is a list of words within Hamming distance <= radius
    neighbors = [[] for _ in range(total_words)]
    
    # Helper to convert int to tuple
    def int_to_tuple(x):
        res = []
        for _ in range(n):
            res.append(x % q)
            x //= q
        return tuple(res)
        
    # Helper to convert tuple to int
    def tuple_to_int(t):
        x = 0
        for i in range(n):
            x = x * q + t[i]
        return x
        
    # Precompute all tuples and their indices
    all_tuples = [int_to_tuple(i) for i in range(total_words)]
    
    for i in range(total_words):
        t = all_tuples[i]
        neighbors[i].append(i) # distance 0
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == t[pos]: continue
                    new_t = list(t)
                    new_t[pos] = val
                    new_i = tuple_to_int(tuple(new_t))
                    neighbors[i].append(new_i)
                    
    # 1. Greedy Construction
    uncovered = set(range(total_words))
    C = []
    
    while uncovered:
        best_c = -1
        best_cover_count = -1
        # To speed up, we can sample or just iterate all if total_words is small (729)
        # 729 is small, so iterate all
        for c in range(total_words):
            covered = 0
            for w in neighbors[c]:
                if w in uncovered:
                    covered += 1
            if covered > best_cover_count:
                best_cover_count = covered
                best_c = c
                if covered == len(uncovered): # Perfect cover found? Unlikely for radius 1 n 6
                    break
        
        if best_c == -1: break
        
        C.append(best_c)
        # Remove covered from uncovered
        for w in neighbors[best_c]:
            if w in uncovered:
                uncovered.remove(w)
                
    # 2. Local Search Refinement
    # Maintain counts array
    counts = [0] * total_words
    for c in C:
        for w in neighbors[c]:
            counts[w] += 1
            
    # Function to check if removing c is valid
    def can_remove(c):
        for w in neighbors[c]:
            if counts[w] <= 1:
                return False
        return True
        
    # Function to apply removal
    def do_remove(c):
        for w in neighbors[c]:
            counts[w] -= 1
        C.remove(c)
            
    # Function to apply swap
    def do_swap(c_old, c_new):
        # Remove old
        for w in neighbors[c_old]:
            counts[w] -= 1
        # Add new
        for w in neighbors[c_new]:
            counts[w] += 1
        C.remove(c_old)
        C.append(c_new)
        
    # Main loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant codewords
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            # Try to find a removable codeword
            # Randomize order to avoid bias? Or just iterate.
            for c in list(C):
                if can_remove(c):
                    do_remove(c)
                    improved = True
                    removed_any = True
                    break # Restart check as counts changed
        
        # Phase 2: Swaps
        if not improved and time.time() < deadline:
            best_swap = None
            best_score = None
            
            # Current score: number of words with count == 1
            current_critical = sum(1 for w in range(total_words) if counts[w] == 1)
            
            for c_old in C:
                # Pre-compute what counts would be if c_old removed
                # To save time, we can just simulate for each c_new
                
                for c_new in range(total_words):
                    if c_new in C:
                        continue
                    
                    # Check validity: no word in S(c_old) should drop to 0
                    # We need to check min count in S(c_old) after removing c_old
                    # Actually, we can just check if min(counts[w] for w in S(c_old)) >= 1
                    # If min is 1, then removing c_old makes it 0 -> Invalid unless c_new covers it
                    # But c_new is added.
                    
                    # Let's simulate counts change
                    # delta counts
                    # For w in S(c_old): count - 1
                    # For w in S(c_new): count + 1
                    
                    # Check if any w in S(c_old) \ S(c_new) drops to 0
                    # And any w in S(c_new) \ S(c_old) increases (good)
                    # And intersection stays same or +0? No, -1+1=0 net change.
                    
                    min_count_after_remove = 999
                    for w in neighbors[c_old]:
                        # Count of w if c_old removed
                        c_w = counts[w] - 1
                        if c_w < min_count_after_remove:
                            min_count_after_remove = c_w
                        if c_w < 0:
                            break
                    if min_count_after_remove < 0:
                        continue # c_old was the only cover for some word not covered by c_new? 
                    # Wait, c_new might cover that word.
                    # So we must check final count.
                    
                    # Efficient check:
                    # For each w in S(c_old):
                    #   final_count = counts[w] - 1 + (1 if w in S(c_new) else 0)
                    #   if final_count < 1: invalid
                    
                    valid = True
                    new_critical = 0
                    # We only need to update critical count for words in S(c_old) U S(c_new)
                    
                    # To speed up, we can just check validity first
                    for w in neighbors[c_old]:
                        new_count = counts[w] - 1
                        # Check if w is in S(c_new)
                        # Precomputed neighbors check is O(1) if we use sets, but lists are O(k)
                        # Let's use a boolean array for S(c_new) for speed?
                        # Or just check intersection.
                        
                        # Simple approach:
                        # If w is in S(c_new), new_count += 1
                        # Since n=6, radius=1, S(c) has 13 elements.
                        # Checking if w in S(c_new) by iterating S(c_new) is 13 ops.
                        # 13 * 13 = 169 ops per pair.
                        # 75 * 729 * 169 ~ 9 million ops per full pass.
                        # 9 million is fast in Python? Maybe 1-2 seconds.
                        # So we can do a few passes.
                        
                        if w in neighbors[c_new]: # This is O(1) if neighbors[c_new] is a set, but it's a list.
                            # Let's convert neighbors to sets for lookup?
                            # Or just use the fact that we can check distance.
                            pass
                    
                    # Let's optimize neighbor lookup
                    # We'll stick to the list check for simplicity first, but maybe convert to set inside loop if needed.
                    # Actually, creating a set for S(c_new) inside the inner loop is expensive.
                    # Let's precompute neighbor sets?
                    # neighbors_set = [set(neb) for neb in neighbors]
                    
                    # Let's refactor to use sets for neighbors for O(1) lookup.
                    # I will do this in the final code.
                    
                    if not valid: continue
                    
                    # Calculate new critical count
                    # Words in S(c_old) \ S(c_new): count decreases by 1
                    # Words in S(c_new) \ S(c_old): count increases by 1
                    # Words in Intersection: count unchanged
                    # Words outside: unchanged
                    
                    # We can just sum up critical counts for the affected words
                    # But we need to know the state of ALL words to count total criticals?
                    # No, we can maintain a global count of criticals and update it incrementally.
                    
                    # Let's just calculate the change in criticals for the union of spheres.
                    # affected = set(neighbors[c_old]) | set(neighbors[c_new])
                    # For each w in affected:
                    #   old_count = counts[w]
                    #   new_count = old_count - (1 if w in S(c_old) else 0) + (1 if w in S(c_new) else 0)
                    #   delta_critical = (1 if new_count == 1 else 0) - (1 if old_count == 1 else 0)
                    #   sum_delta += delta_critical
                    # new_total_critical = current_critical + sum_delta
                    
                    # If new_total_critical < best_score, update best.
                    # If equal, maybe pick random?
                    
                    pass # Placeholder for logic
            
            # Apply best swap if found
            if best_swap:
                do_swap(best_swap[0], best_swap[1])
                improved = True

    return [list(all_tuples[c]) for c in C]
