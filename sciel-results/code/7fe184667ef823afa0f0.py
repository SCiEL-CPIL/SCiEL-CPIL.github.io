import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Precompute conflicts for fast can_add check
    # For each x, store a list of (y, z) pairs such that {x, y, z} is a 3-AP
    # We need to check if y and z are both in S.
    # We will use bitmasks for S.
    
    # conflict_pairs[x] is a list of (y, z)
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s = a + b
            if s % 2 == 0:
                c = s // 2
                if c > b:
                    # a, b, c is a 3-AP
                    # For a: pairs with b, c
                    conflict_pairs[a].append((b, c))
                    # For b: pairs with a, c
                    conflict_pairs[b].append((a, c))
                    # For c: pairs with a, b
                    conflict_pairs[c].append((a, b))
                elif c == b:
                    # a, c, c? No, distinct terms required for nonconstant AP?
                    # "nonconstant k=3 term arithmetic progression"
                    # Usually implies distinct terms. a < b < c.
                    pass
    
    def can_add_bitset(x, S_mask):
        """Check if x can be added to set represented by S_mask."""
        for y, z in conflict_pairs[x]:
            # Check if y and z are both in S_mask
            if (S_mask & (1 << y)) and (S_mask & (1 << z)):
                return False
        return True

    def get_float_sum(S_mask):
        """Compute floating-point reciprocal sum from bitmask."""
        total = 0.0
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                total += recip[x]
        return total

    def greedy_fill(forbidden_mask, S_mask):
        """Fill S by adding elements in ascending order."""
        S = S_mask
        for x in all_nums:
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add_bitset(x, S):
                S |= (1 << x)
        return S

    def greedy_fill_descending(forbidden_mask, S_mask):
        """Fill S by adding elements in descending order."""
        S = S_mask
        for x in reversed(all_nums):
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add_bitset(x, S):
                S |= (1 << x)
        return S

    def local_search(S_mask, max_time):
        """Run local search starting from S_mask, return best mask found."""
        S = S_mask
        sum_float = get_float_sum(S)
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # Strategy: 1-Swap
            # Try replacing x in S with y not in S
            # We want to maximize gain: recip[y] - recip[x]
            # So we prefer small y and large x.
            
            # Iterate x in S (largest first to try to remove big weights? No, we want to remove big x to replace with small y)
            # Actually, we want to replace x with y if 1/y > 1/x, i.e., y < x.
            # So we should try to find small y that can replace large x.
            
            found_swap = False
            best_gain = 0
            best_x = -1
            best_y = -1
            
            # To make this faster, we can iterate y in ascending order (small y first) and x in descending order?
            # Or just iterate all pairs. N=256, so 256*256 = 65536 checks per pass.
            # Each check is O(degree) which is small.
            # This is very fast.
            
            # We will iterate y (candidate to add) in ascending order (1 to N)
            # And x (candidate to remove) in descending order (N to 1)
            # If we find a valid swap with positive gain, we take it.
            # To ensure we find the *best* swap, we might need to check all, but taking the first valid positive gain is often sufficient for local search progress.
            # Let's try to find the best swap in one pass to be more rigorous.
            
            for y in range(1, N + 1):
                if (time.time() - start_time) > max_time:
                    break
                if S & (1 << y):
                    continue
                
                for x in range(N, 0, -1):
                    if (time.time() - start_time) > max_time:
                        break
                    if not (S & (1 << x)):
                        continue
                    
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        continue # No gain or loss, skip to avoid cycles
                    
                    # Check if removing x and adding y is valid
                    S_new = S & ~(1 << x)
                    if can_add_bitset(y, S_new):
                        # Valid swap
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
                            # We can't break immediately if we want the best, but for speed, 
                            # taking the first good one is often fine. 
                            # However, a "greedy" local search usually takes the best move.
                            # Let's just take the first valid move to keep it fast, 
                            # or break if we find a very good one?
                            # Let's stick to finding the best in the pass.
            
            if best_y != -1:
                S = S & ~(1 << best_x)
                S |= (1 << best_y)
                sum_float += best_gain
                improved = True
                continue
            
            # If no 1-swap worked, try 1-Remove (and re-greedy)
            # Remove an element x and re-greedy fill the rest.
            # This allows adding multiple small elements.
            
            best_remove_gain = -1
            best_remove_x = -1
            
            for x in range(N, 0, -1):
                if (time.time() - start_time) > max_time:
                    break
                if not (S & (1 << x)):
                    continue
                
                S_no_x = S & ~(1 << x)
                # Re-greedy from S_no_x
                S_new = greedy_fill(1 << x, S_no_x) # Forbid x
                sum_new = get_float_sum(S_new)
                gain = sum_new - sum_float
                
                if gain > 1e-15 and gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove_x = x
                    best_remove_S = S_new
            
            if best_remove_x != -1:
                S = best_remove_S
                sum_float += best_remove_gain
                improved = True
                continue
            
            # If still stuck, try adding a small element if possible (shouldn't happen if maximal, but check)
            added = False
            for y in range(1, N + 1):
                if (time.time() - start_time) > max_time:
                    break
                if S & (1 << y):
                    continue
                if can_add_bitset(y, S):
                    S |= (1 << y)
                    sum_float += recip[y]
                    improved = True
                    added = True
                    break
            
            if not added:
                # Try a random perturbation: remove a random element and re-greedy
                if (time.time() - start_time) > max_time:
                    break
                random_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                if not random_elements:
                    break
                x = random.choice(random_elements)
                S_no_x = S & ~(1 << x)
                S_new = greedy_fill(1 << x, S_no_x)
                sum_new = get_float_sum(S_new)
                if sum_new > sum_float:
                    S = S_new
                    sum_float = sum_new
                    improved = True
        
        return S, sum_float

    # Step 1: Initializations
    initializations = []
    
    # Greedy ascending
    S_asc = greedy_fill(0, 0)
    initializations.append(S_asc)
    
    # Greedy descending
    S_desc = greedy_fill_descending(0, 0)
    initializations.append(S_desc)
    
    # Randomized greedy
    for trial in range(10):
        if (time.time() - start_time) > deadline - 20:
            break
        random.seed(trial * 12345 + 67890)
        S_rand = 0
        for x in all_nums:
            # Randomly skip with small probability
            if random.random() < 0.1 and x > 5:
                continue
            if can_add_bitset(x, S_rand):
                S_rand |= (1 << x)
        initializations.append(S_rand)
    
    # Step 2: Local Search
    best_S = 0
    best_sum = 0.0
    
    time_per_ls = max(5.0, (deadline - 10 - (time.time() - start_time)) / max(1, len(initializations)))
    
    for i, S_init in enumerate(initializations):
        if (time.time() - start_time) > deadline - 5:
            break
        
        remaining_time = deadline - 5 - (time.time() - start_time)
        if remaining_time < 3:
            break
        
        S_ls, sum_ls = local_search(S_init, min(time_per_ls, remaining_time))
        
        if sum_ls > best_sum:
            best_sum = sum_ls
            best_S = S_ls
    
    # Step 3: Final Polish
    if best_S != 0:
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining > 5:
            S_final, sum_final = local_search(best_S, remaining)
            if sum_final > best_sum:
                best_S = S_final
                best_sum = sum_final
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if best_S & (1 << x)]
    
    # Validate
    S_set = set(result)
    valid = True
    for a in S_set:
        for b in S_set:
            if a >= b:
                continue
            s = a + b
            if s % 2 == 0:
                mid = s // 2
                if mid in S_set and mid != a and mid != b:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid set if validation fails (should not happen)
        # Use greedy result
        result = [x for x in range(1, N + 1) if S_asc & (1 << x)]
        
    return result
