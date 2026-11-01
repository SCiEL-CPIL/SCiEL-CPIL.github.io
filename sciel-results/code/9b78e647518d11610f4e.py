import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Optimized for the specific case q=3, n=6, radius=1 (K_3(6,1)).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    total_words = q ** n
    
    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors for all words as lists
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val

    # Build neighbor bitmasks for fast coverage checks
    # Use Python integers as bitmasks (arbitrary precision)
    def build_mask(indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        return mask

    neighbor_masks = [build_mask(neighbors[i]) for i in range(total_words)]
    full_mask = (1 << total_words) - 1

    # --- Phase 1: Greedy Initialization ---
    def greedy_init():
        uncovered = full_mask
        codewords = set()
        while uncovered:
            best_center = -1
            best_count = 0
            # Randomize order slightly to break symmetries? No, greedy is deterministic.
            # We can sample a subset if it's too slow, but 729 is small.
            for c in range(total_words):
                if c in codewords:
                    continue
                count = bin(uncovered & neighbor_masks[c]).count('1')
                if count > best_count:
                    best_count = count
                    best_center = c
            if best_center == -1:
                break
            codewords.add(best_center)
            uncovered &= ~neighbor_masks[best_center]
        return codewords

    # --- Phase 2: Reduction ---
    def reduce_codewords(codewords, time_limit=1.0):
        """
        Removes redundant codewords.
        A codeword c is redundant if coverage(codewords - {c}) covers neighbors(c).
        """
        if time.time() > time_limit:
            return codewords
            
        # Calculate initial coverage
        coverage = 0
        for c in codewords:
            coverage |= neighbor_masks[c]
        
        changed = True
        while changed:
            if time.time() > time_limit:
                break
            changed = False
            # Try to remove each codeword
            # Optimization: Check in random order to avoid bias
            candidates = list(codewords)
            random.shuffle(candidates)
            for c in candidates:
                if c not in codewords:
                    continue
                # Coverage without c
                cover_without_c = coverage & ~neighbor_masks[c]
                # Check if neighbors of c are still covered
                if (cover_without_c & neighbor_masks[c]) == neighbor_masks[c]:
                    codewords.remove(c)
                    coverage = cover_without_c
                    changed = True
                    # Break to restart loop with new set? 
                    # Or continue? Restarting is safer for correctness of 'coverage' variable 
                    # if we don't update it carefully. Here we update coverage.
                    # However, removing one might make another redundant that wasn't before?
                    # No, removing codewords only shrinks coverage. 
                    # If c was redundant, removing it doesn't uncover anything.
                    # So we can continue checking other candidates with the updated coverage.
        return codewords

    # --- Phase 3: Swap Local Search ---
    def local_search(codewords, time_limit=1.0):
        """
        Tries to swap codewords to find a smaller set.
        This is more complex. A simpler approach: 
        If we are stuck at size K, try removing a random word, then repair.
        """
        current_size = len(codewords)
        best_codewords = set(codewords)
        best_size = current_size
        
        while time.time() < time_limit:
            if len(best_codewords) < best_size:
                best_size = len(best_codewords)
            
            # Perturbation: Remove 1-3 random codewords
            k = random.randint(1, 3)
            if len(best_codewords) <= k:
                break
            to_remove = random.sample(list(best_codewords), k)
            perturbed = set(best_codewords)
            for c in to_remove:
                perturbed.remove(c)
            
            # Repair: Greedy add
            coverage = 0
            for c in perturbed:
                coverage |= neighbor_masks[c]
            uncovered = full_mask & ~coverage
            
            while uncovered:
                if time.time() > time_limit:
                    break
                best_center = -1
                best_count = 0
                # Only consider candidates not already in perturbed
                # To speed up, we can just scan all, 729 is small
                for c in range(total_words):
                    if c in perturbed:
                        continue
                    count = bin(uncovered & neighbor_masks[c]).count('1')
                    if count > best_count:
                        best_count = count
                        best_center = c
                if best_center == -1:
                    break
                perturbed.add(best_center)
                uncovered &= ~neighbor_masks[best_center]
            
            # Reduce the repaired set
            perturbed = reduce_codewords(perturbed, time_limit=0.5)
            
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_codewords = perturbed
        
        return best_codewords

    # --- Main Loop ---
    
    # Start with a good initial solution
    # The direct sum of tetracode gives 73 words.
    # We can try to construct it or just use greedy.
    # Greedy usually finds 73-75.
    current_codewords = greedy_init()
    current_codewords = reduce_codewords(current_codewords, time_limit=2.0)
    
    best_codewords = set(current_codewords)
    best_size = len(best_codewords)
    
    # Iterated Local Search
    while time.time() < deadline:
        # 1. Reduce current
        current_codewords = reduce_codewords(current_codewords, time_limit=1.0)
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            # If we hit 71, we can stop early? 
            # 71 is the lower bound. If we find 71, it's optimal.
            if best_size == 71:
                break

        # 2. Local Search / Perturbation
        # Use local search which includes perturbation and repair
        # We give it a time slice
        time_slice = 5.0
        if time.time() + time_slice > deadline:
             time_slice = deadline - time.time()
        
        if time_slice > 0.5:
             new_codewords = local_search(current_codewords, time_limit=time_slice)
             current_codewords = new_codewords
             
        # If local search didn't improve, do a simple kick
        # Remove 2-4 random words and repair
        if random.random() < 0.5:
             k = random.randint(2, 4)
             if len(current_codewords) > k:
                 to_remove = random.sample(list(current_codewords), k)
                 for c in to_remove:
                     current_codewords.remove(c)
                 # Quick repair
                 coverage = 0
                 for c in current_codewords:
                     coverage |= neighbor_masks[c]
                 uncovered = full_mask & ~coverage
                 while uncovered and time.time() < deadline:
                     best_center = -1
                     best_count = 0
                     for c in range(total_words):
                         if c in current_codewords:
                             continue
                         count = bin(uncovered & neighbor_masks[c]).count('1')
                         if count > best_count:
                             best_count = count
                             best_center = c
                     if best_center == -1:
                         break
                     current_codewords.add(best_center)
                     uncovered &= ~neighbor_masks[best_center]
                 
                 # Reduce again
                 current_codewords = reduce_codewords(current_codewords, time_limit=1.0)
                 
                 if len(current_codewords) < best_size:
                     best_size = len(current_codewords)
                     best_codewords = set(current_codewords)
                     if best_size == 71:
                         break

    # Final polish: one more rigorous reduction pass on the best found
    if time.time() < deadline - 1.0:
        final_codewords = set(best_codewords)
        final_codewords = reduce_codewords(final_codewords, time_limit=deadline - time.time())
        if len(final_codewords) < best_size:
            best_codewords = final_codewords

    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result

if __name__ == "__main__":
    # Test run
    # q, n, radius = 3, 6, 1
    # res = construct(q, n, radius)
    # print(len(res))
    pass
