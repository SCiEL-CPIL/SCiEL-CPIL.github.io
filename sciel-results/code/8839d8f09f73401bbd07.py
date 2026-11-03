import time
import sys

def construct(q, n, radius):
    # Deadline guard
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Handle specific case q=3, n=7, radius=1 with optimized bitmask approach
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n
        full_mask = (1 << N) - 1
        
        # Precompute balls
        # word i is a number in base 3.
        # To get digits of i:
        def get_digits(i, n, q):
            return [int((i // (q ** k)) % q) for k in range(n)]
        
        balls = [0] * N
        for i in range(N):
            mask = 0
            # distance 0
            mask |= (1 << i)
            # distance 1
            digits = get_digits(i, n, q)
            for pos in range(n):
                orig_val = digits[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    # create new word index
                    # subtract orig_val * 3^pos, add new_val * 3^pos
                    diff = (new_val - orig_val) * (q ** pos)
                    j = i + diff
                    mask |= (1 << j)
            balls[i] = mask
            
        # Greedy Set Cover
        uncovered = full_mask
        sol = []
        sol_set = set()
        
        # Optimization: list of all indices
        all_indices = list(range(N))
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_count = -1
            
            # Find best word to cover most uncovered
            # We can iterate all, or just those not in sol.
            # Since N=2187, iterating all is fine.
            for i in all_indices:
                if i in sol_set:
                    continue
                # Count newly covered
                # balls[i] & uncovered
                cnt = (balls[i] & uncovered).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1:
                break
                
            sol.append(best_idx)
            sol_set.add(best_idx)
            uncovered &= ~balls[best_idx]
            
        # Local Search
        # We maintain sol_mask for quick checks? 
        # Actually, recalculating sol_mask for removal/swap checks might be expensive if done naively.
        # But N is small.
        # Let's keep sol as list, and sol_set for membership.
        # For coverage check, we can maintain a bitmask `current_cover`.
        
        current_cover = 0
        for i in sol:
            current_cover |= balls[i]
            
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Removal
            new_sol = []
            for i in sol:
                # Check if i is redundant
                # If (current_cover without i) still covers full_mask
                # current_cover_without_i = current_cover & ~balls[i]
                # This is NOT correct. current_cover is union of all.
                # We need to know if elements of balls[i] are covered by others.
                # Let C_others = current_cover & ~balls[i] ? No.
                # C_others is the union of all balls except i.
                # We don't have C_others directly.
                # However, if (current_cover & ~balls[i]) == full_mask, it means 
                # every bit in full_mask is set in current_cover AND is NOT in balls[i]?
                # No. 
                # Condition for removal: The union of all other sets must be full_mask.
                # Let U = full_mask.
                # We need (Union_{j != i} balls[j]) == U.
                # Note that Union_{j != i} balls[j] is a subset of current_cover.
                # Actually, Union_{j != i} balls[j] = current_cover \ (balls[i] \ Union_{j != i} balls[j])?
                # Simpler: 
                # If we remove i, new cover is C' = (current_cover & ~balls[i])? 
                # NO. current_cover is OR of all. 
                # C' is OR of all except i.
                # C' is NOT simply current_cover & ~balls[i]. 
                # Example: A={1,2}, B={2,3}. C={1,2,3}. 
                # Remove A. C'={2,3}.
                # current_cover & ~balls[A] = {1,2,3} & ~{1,2} = {3}. Incorrect.
                
                # So we must calculate the coverage of the other sets.
                # Since sol size is small (~200), and N is 2187, 
                # calculating union of 199 masks is fast.
                # But doing it for every i in every pass is O(|sol|^2 * cost_union).
                # Cost union is just bitwise OR.
                
                # Alternative: 
                # We can check if balls[i] is subset of (Union of others).
                # This is equivalent to: 
                # Let C_others = 0
                # for j in sol: if j != i: C_others |= balls[j]
                # if C_others == full_mask: remove i.
                
                # This is O(|sol|) per i. Total O(|sol|^2) per pass.
                # |sol| ~ 200. 40,000 ORs. Very fast.
                
                C_others = 0
                for j in sol:
                    if j != i:
                        C_others |= balls[j]
                
                if C_others == full_mask:
                    # i is redundant
                    sol.remove(i)
                    sol_set.remove(i)
                    # Recalculate current_cover
                    current_cover = 0
                    for j in sol:
                        current_cover |= balls[j]
                    improved = True
                    break # Restart pass as sol changed
                
            # 2. Swaps
            # If we broke from removal, continue outer loop
            if improved:
                continue
                
            # Try swaps
            # For each i in sol, for each j not in sol
            # Check if (Union_{k in sol, k!=i} balls[k]) | balls[j] == full_mask
            
            # Precompute "coverage without i" for each i?
            # That's expensive.
            # Let's just iterate.
            
            swapped = False
            for i in list(sol): # iterate copy
                # Calculate C_others for i
                C_others = 0
                for j in sol:
                    if j != i:
                        C_others |= balls[j]
                
                # Find j not in sol such that C_others | balls[j] == full_mask
                # This means balls[j] must cover (full_mask & ~C_others)
                # Let missing = full_mask & ~C_others
                missing = full_mask & ~C_others
                
                if missing == 0:
                    # i is redundant, should have been caught above?
                    # Maybe due to order?
                    # Actually if missing is 0, we can just remove i.
                    # But we already checked removals.
                    pass
                
                # Look for j
                # We can iterate all j not in sol
                for j in all_indices:
                    if j in sol_set:
                        continue
                    # Check if balls[j] covers missing
                    if (balls[j] & missing) == missing:
                        # Valid swap
                        sol.remove(i)
                        sol_set.remove(i)
                        sol.append(j)
                        sol_set.add(j)
                        
                        # Update current_cover
                        current_cover = 0
                        for k in sol:
                            current_cover |= balls[k]
                        
                        improved = True
                        swapped = True
                        break # Break inner j loop
                
                if swapped:
                    break # Break outer i loop, restart local search pass
            
        # Convert indices to words
        result = []
        for idx in sol:
            word = get_digits(idx, n, q)
            result.append(word)
            
        return result

    else:
        # Generic fallback: Simple Greedy
        # This might be slow for large n, but correct.
        N = q ** n
        full_mask = (1 << N) - 1
        
        def get_digits(i, n, q):
            return [int((i // (q ** k)) % q) for k in range(n)]
        
        balls = [0] * N
        for i in range(N):
            mask = 0
            digits = get_digits(i, n, q)
            # Generate all neighbors within radius
            # This is expensive for large n/radius.
            # For fallback, we assume small N.
            import itertools
            for delta in itertools.product(range(-radius, radius+1), repeat=n):
                # Check if L1 norm <= radius
                if sum(abs(d) for d in delta) <= radius:
                    # Check bounds
                    valid = True
                    new_digits = []
                    for k in range(n):
                        val = digits[k] + delta[k]
                        if val < 0 or val >= q:
                            valid = False
                            break
                        new_digits.append(val)
                    if valid:
                        # Convert to index
                        j = 0
                        for k, v in enumerate(new_digits):
                            j += v * (q ** k)
                        mask |= (1 << j)
            balls[i] = mask
            
        uncovered = full_mask
        sol = []
        sol_set = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_count = -1
            for i in range(N):
                if i in sol_set:
                    continue
                cnt = (balls[i] & uncovered).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1:
                break
            
            sol.append(best_idx)
            sol_set.add(best_idx)
            uncovered &= ~balls[best_idx]
            
        result = []
        for idx in sol:
            word = get_digits(idx, n, q)
            result.append(word)
        return result
