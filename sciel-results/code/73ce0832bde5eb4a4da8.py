import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Handle specific case q=3, n=6, radius=1 efficiently
    if q == 3 and n == 6 and radius == 1:
        space_size = q ** n # 729
        
        # Precompute neighbors as bitmasks
        # Index mapping: word w -> index i = sum(w[j] * 3^(5-j)) ? 
        # Let's use standard base-q expansion where index = w[0]*q^(n-1) + ... + w[n-1]
        # But for bitmask, we just need a unique ID 0..728.
        # Let's use the index as the ID.
        
        # Precompute digits for each index to find neighbors quickly?
        # Or just generate neighbors on the fly for precomputation?
        # Precomputation is one-time cost.
        
        neighbors = [0] * space_size
        powers = [q ** i for i in range(n)] # 1, 3, 9, 27, 81, 243
        
        # Map index to word digits
        # idx = d0 * 3^5 + d1 * 3^4 + ... + d5
        # Wait, standard integer representation:
        # idx % 3 = d5 (last digit)
        # (idx // 3) % 3 = d4
        # ...
        # Let's define idx such that word w corresponds to idx = sum(w[i] * 3^(n-1-i))
        
        for idx in range(space_size):
            # Get digits
            digits = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                digits[i] = temp % q
                temp //= q
            
            # Self
            mask = 1 << idx
            
            # Neighbors: change one digit
            for pos in range(n):
                orig_val = digits[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    # Calculate new index
                    # diff = (new_val - orig_val) * 3^(n-1-pos)
                    # new_idx = idx + diff
                    # Since we are in base 3, and we only change one digit, 
                    # and new_val, orig_val in 0..2, the carry/borrow doesn't affect other digits
                    # as long as we treat it as integer arithmetic.
                    # Example: idx = 10 (base 10 analog) -> digits [1, 0]. Change 1->2. New idx = 20.
                    # In base 3: idx=3 (digits [1,0]). Change d0(1) to 2. New idx = 6 (digits [2,0]).
                    # Formula: new_idx = idx + (new_val - orig_val) * (3 ** (n-1-pos))
                    power = 3 ** (n - 1 - pos)
                    new_idx = idx + (new_val - orig_val) * power
                    mask |= (1 << new_idx)
            
            neighbors[idx] = mask
        
        all_mask = (1 << space_size) - 1
        
        best_code = None
        best_size = space_size # Start with worst case
        
        # Run multiple greedy trials
        num_trials = 0
        while time.time() < deadline - 5.0:
            num_trials += 1
            uncovered = all_mask
            code = []
            
            while uncovered:
                # Find best candidate
                max_cov = -1
                best_cands = []
                
                # Optimization: If uncovered is small, we can be faster? 
                # For now, scan all 729 words.
                for w in range(space_size):
                    overlap = (neighbors[w] & uncovered).bit_count()
                    if overlap > max_cov:
                        max_cov = overlap
                        best_cands = [w]
                    elif overlap == max_cov:
                        best_cands.append(w)
                        
                if not best_cands:
                    break # Should not happen if uncovered is not 0
                    
                # Random selection from best candidates to diversify trials
                chosen = random.choice(best_cands)
                code.append(chosen)
                uncovered &= ~neighbors[chosen]
                
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
                
                # If we hit the known lower bound vicinity, we might stop early if we want, 
                # but let's keep searching for a bit to see if we can hit 71 (unlikely) or stabilize at 73.
                if best_size <= 73:
                    # We found a very good solution. Let's do a local search polish.
                    if time.time() < deadline - 10.0:
                        polished_code = local_search_polish(best_code, neighbors, all_mask, space_size, q, n)
                        if len(polished_code) < best_size:
                            best_size = len(polished_code)
                            best_code = polished_code
                        
                        # Continue searching for a few more trials to see if we can beat it
                        if num_trials > 100:
                            break
                    else:
                        break
                        
        # If no code was found (should not happen), fallback
        if not best_code:
            # Fallback to tetracode construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Convert indices back to lists of digits
        res = []
        for idx in best_code:
            word = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                word[i] = temp % q
                temp //= q
            res.append(word)
            
        return res

    # Generic fallback for other parameters (simple greedy)
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    neighbors = []
    for x in range(total_words):
        nb_set = {x}
        for i in range(n):
            digit = (x // powers[i]) % q
            for delta in range(1, radius + 1):
                if digit - delta >= 0:
                    nb_set.add(x - delta * powers[i])
                if digit + delta < q:
                    nb_set.add(x + delta * powers[i])
        mask = 0
        for nb in nb_set:
            mask |= (1 << nb)
        neighbors.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = total_words
    
    for trial in range(100):
        if time.time() > deadline:
            break
        uncovered = all_mask
        code = []
        while uncovered:
            max_cov = -1
            best_cands = []
            for w in range(total_words):
                overlap = (neighbors[w] & uncovered).bit_count()
                if overlap > max_cov:
                    max_cov = overlap
                    best_cands = [w]
                elif overlap == max_cov:
                    best_cands.append(w)
            if not best_cands:
                break
            chosen = random.choice(best_cands)
            code.append(chosen)
            uncovered &= ~neighbors[chosen]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    if best_code:
        res = []
        for idx in best_code:
            word = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                word[i] = temp % q
                temp //= q
            res.append(word)
        return res
    else:
        return [[0]*n]

def local_search_polish(code, neighbors, all_mask, space_size, q, n):
    """
    Attempts to improve the code size via swaps and deletions.
    """
    current_code = code.copy()
    current_set = set(current_code)
    
    # Precompute union of neighbors for current code
    # Union is bitwise OR of all neighbors in code
    union_mask = 0
    for c in current_code:
        union_mask |= neighbors[c]
        
    # If it's not a cover, we can't polish (but greedy ensures it is)
    # Check if cover
    if (union_mask & all_mask) != all_mask:
        return current_code

    improved = True
    while improved:
        improved = False
        
        # 1. Deletion Pass
        # Try to remove any codeword that is redundant
        # A codeword c is redundant if (union_mask ^ neighbors[c]) still covers all_mask
        # i.e., (union_mask & ~neighbors[c]) | neighbors[c] == union_mask is trivial
        # We need: (union_mask_without_c) covers all_mask
        # union_mask_without_c = union_mask & ~neighbors[c] ? NO.
        # If c is the ONLY one covering a word w, then removing c uncovers w.
        # So we check: if (union_mask & ~neighbors[c]) | neighbors[c] ... wait.
        # The coverage provided by the set C is Union(N_c for c in C).
        # If we remove c, the new coverage is Union(N_k for k in C\{c}).
        # We don't have this value directly.
        # However, we can check if there exists a word w such that N_c is the ONLY neighbor set covering w.
        # This is equivalent to: Is there a w in all_mask such that w is in N_c and w is NOT in Union(N_k for k in C\{c})?
        # This is hard to compute efficiently without maintaining counts or the union incrementally.
        
        # Simpler Deletion Check:
        # For each c in current_code:
        #   temp_union = 0
        #   for k in current_code:
        #       if k != c: temp_union |= neighbors[k]
        #   if (temp_union & all_mask) == all_mask:
        #       # c is redundant
        #       current_code.remove(c)
        #       current_set.remove(c)
        #       union_mask = temp_union
        #       improved = True
        #       break # Restart deletion pass
        
        # This is O(N^2 * BitOp). N~73. 73*73 = 5329 bit ops. Very fast.
        
        found_redundant = False
        for c in current_code:
            temp_union = 0
            for k in current_code:
                if k != c:
                    temp_union |= neighbors[k]
            if (temp_union & all_mask) == all_mask:
                current_code.remove(c)
                current_set.remove(c)
                union_mask = temp_union
                found_redundant = True
                break
        
        if found_redundant:
            improved = True
            continue
            
        # 2. Swap Pass
        # Try to swap c in current_code with c' not in current_code
        # New code C' = (C \ {c}) ∪ {c'}
        # New Union = (Union(C\{c}) | N_{c'})
        # We need New Union to cover all_mask.
        
        # To make this efficient, we can precompute Union(C\{c}) for all c?
        # That's O(N^2) space/time.
        # Or just do it on the fly for a limited number of swaps.
        
        # Let's try a few random swaps
        swaps_made = 0
        max_swaps_to_try = 50
        
        for _ in range(max_swaps_to_try):
            if len(current_code) == 0: break
            c = random.choice(current_code)
            
            # Calculate Union(C\{c})
            union_without_c = 0
            for k in current_code:
                if k != c:
                    union_without_c |= neighbors[k]
            
            # Find a c' that covers the holes
            # Holes = all_mask & ~union_without_c
            holes = all_mask & ~union_without_c
            
            if holes == 0:
                # c is redundant (should have been caught above, but maybe race condition?)
                current_code.remove(c)
                current_set.remove(c)
                union_mask = union_without_c
                improved = True
                break
            
            # We need N_{c'} to cover 'holes'
            # So we look for c' not in current_set such that (N_{c'} & holes) == holes
            # i.e., N_{c'} is a superset of holes.
            
            # Iterate through all possible c'? 729 checks.
            # Check if (neighbors[c'] & holes) == holes
            # And c' not in current_set
            
            # Optimization: Only check c' that are in 'holes' or their neighbors?
            # Actually, c' must cover all words in 'holes'.
            # So c' must be within distance 1 of all words in 'holes'.
            # This implies c' is a specific set of candidates.
            
            # Simple scan:
            candidate_found = False
            for c_prime in range(space_size):
                if c_prime in current_set:
                    continue
                if (neighbors[c_prime] & holes) == holes:
                    # Valid swap
                    # Update state
                    current_code.remove(c)
                    current_set.remove(c)
                    current_code.append(c_prime)
                    current_set.add(c_prime)
                    
                    # Update union_mask
                    union_mask = union_without_c | neighbors[c_prime]
                    
                    improved = True
                    swaps_made += 1
                    candidate_found = True
                    break
            
            if candidate_found:
                break # Restart swap pass after a successful swap
        
    return current_code
