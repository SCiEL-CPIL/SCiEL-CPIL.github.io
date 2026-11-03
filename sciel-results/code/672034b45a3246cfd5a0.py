import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    limit = 145.0 # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute powers of q for index conversion
    powers = [q**i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor masks
    # mask[idx] is an integer with bits set for all words within radius of idx
    neighbor_masks = [0] * total_words
    
    for idx in range(total_words):
        mask = 1 << idx
        # Radius 1 neighbors
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    mask |= (1 << neighbor_idx)
        neighbor_masks[idx] = mask
        
    # Greedy Set Cover using bitmasks
    uncovered = (1 << total_words) - 1
    solution_indices = []
    
    while uncovered:
        if time.time() - start_time > limit:
            break
            
        best_word = -1
        max_covered = 0
        
        # Find the word that covers the most uncovered words
        for w in range(total_words):
            count = (neighbor_masks[w] & uncovered).bit_count()
            if count > max_covered:
                max_covered = count
                best_word = w
                
        if best_word == -1:
            break
            
        solution_indices.append(best_word)
        uncovered &= ~neighbor_masks[best_word]
        
    # Local Search with Swaps and Removals
    new_solution = list(solution_indices)
    
    # Helper to check if a set of indices is a valid cover
    # We will maintain coverage counts dynamically for speed
    
    # Initialize coverage counts for the current solution
    coverage_count = [0] * total_words
    for w in new_solution:
        m = neighbor_masks[w]
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            coverage_count[idx] += 1
            m ^= lsb
            
    improved = True
    while improved and (time.time() - start_time < limit):
        improved = False
        
        # Phase 1: Remove redundant codewords
        for i in range(len(new_solution) - 1, -1, -1):
            if time.time() - start_time > limit:
                return [get_digits(idx) for idx in new_solution]
            
            w = new_solution[i]
            mask = neighbor_masks[w]
            redundant = True
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                if coverage_count[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
                
            if redundant:
                # Remove w
                new_solution.pop(i)
                # Update coverage counts
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    coverage_count[idx] -= 1
                    m ^= lsb
                improved = True
                break # Restart loop to re-evaluate from scratch or continue? 
                # Breaking and restarting the outer while loop ensures we don't miss new redundancies created by previous removals in the same pass?
                # Actually, removing a redundant word doesn't make other words non-redundant, it might make them redundant if they were only covered by this one? No, if w was redundant, its neighbors were covered by >= 2. Removing it reduces count to >= 1. It doesn't break coverage.
                # So we can continue the loop to find more redundancies in the same pass.
                # But wait, if we don't break, we continue checking i-1. This is fine.
        
        if improved:
            continue # Go back to start of while loop to check for more removals
        
        # Phase 2: Try swaps
        # For each w in solution, try to swap with w' not in solution
        sol_set = set(new_solution)
        for i in range(len(new_solution)):
            if time.time() - start_time > limit:
                return [get_digits(idx) for idx in new_solution]
                
            w = new_solution[i]
            mask_w = neighbor_masks[w]
            
            # Check if w is "essential" for any word
            # If there is a word x covered ONLY by w, w cannot be swapped/removed.
            # This is already checked in Phase 1. If no redundancy, every word in mask_w has count >= 2.
            
            # Try swapping w with some w'
            # To be efficient, we can look for w' that covers the "unique" parts of w?
            # Actually, just iterate through all w' not in sol_set.
            # Optimization: Only try w' that are "similar" to w? Or just all.
            # 6561 * 400 = 2.6M checks. Each check is ~17 bit ops. ~40M ops. Should be < 1s.
            
            found_swap = False
            for w_prime in range(total_words):
                if w_prime in sol_set:
                    continue
                    
                mask_wp = neighbor_masks[w_prime]
                
                # Check if swap is valid
                # Condition: For all x in N(w) \ N(w'), coverage_count[x] >= 2
                # (Because we remove w, so count drops by 1, and w' doesn't cover it, so it doesn't go up)
                
                # Intersection of mask_w and ~mask_wp
                # We want to check bits set in mask_w but not in mask_wp.
                diff_mask = mask_w & ~mask_wp
                
                valid = True
                m = diff_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if coverage_count[idx] < 2:
                        valid = False
                        break
                    m ^= lsb
                    
                if valid:
                    # Perform swap
                    # Remove w
                    m = mask_w
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] -= 1
                        m ^= lsb
                        
                    # Add w_prime
                    m = mask_wp
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] += 1
                        m ^= lsb
                        
                    new_solution[i] = w_prime
                    sol_set.remove(w)
                    sol_set.add(w_prime)
                    found_swap = True
                    improved = True
                    break # Break out of w_prime loop, restart outer loops?
                    # Breaking w_prime loop. We should probably restart the whole local search
                    # to see if this swap enables further removals or swaps.
            
            if found_swap:
                break # Break out of i loop, restart while loop
        
    return [get_digits(idx) for idx in new_solution]
