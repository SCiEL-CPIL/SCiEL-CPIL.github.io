import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 1
    best_len = 1
    
    # Two-pointer sweep: for each right endpoint r, maintain the minimum valid left endpoint
    # An interval [l, r] is valid if for all x in [l, r], inv[x] is not in [l, r]
    # and x != inv[x] (self-inverse check)
    
    # For a fixed r, the constraint is:
    # 1. r != inv[r] (r is not self-inverse)
    # 2. For all x in [l, r], inv[x] < l or inv[x] > r
    #    Equivalently, l > max{inv[x] : x in [l, r] and inv[x] <= r}
    #    and l > max{x : x in [l, r] and inv[x] <= r and inv[x] >= l} -- this is circular
    
    # Better approach: For each r, find the maximum l such that [l, r] is valid.
    # A conflict occurs if there exists x in [l, r] with inv[x] in [l, r].
    # This means l must be greater than any x in [l, r] whose inverse is in [l, r],
    # AND l must be greater than any inv[x] for x in [l, r] where inv[x] is in [l, r].
    
    # Key insight: For a fixed r, define conflict_l(r) as the set of all values v such that
    # there exists x with min(x, inv[x]) <= v <= max(x, inv[x]) and max(x, inv[x]) <= r.
    # Actually, let's think differently.
    
    # For each pair (a, b) where b = inv[a] and a != b, the interval [l, r] is invalid if
    # it contains both a and b. So l > max(a,b) or r < min(a,b).
    # Since we fix r, we need l > max(a,b) for all pairs with min(a,b) <= r and max(a,b) <= r.
    # Wait, that's not quite right either. The interval [l, r] contains both a and b iff
    # l <= min(a,b) and max(a,b) <= r.
    # So for fixed r, the constraint is: l > min(a,b) for all pairs (a,b) with max(a,b) <= r.
    # Also, l must be such that no self-inverse element is in [l, r].
    
    # Let's define for each r:
    #   min_conflict[r] = max{min(a, inv[a]) : a in [1, r], inv[a] <= r, a != inv[a]}
    #   Actually we need: for all a in [l, r], inv[a] not in [l, r].
    #   If inv[a] < l, that's fine. If inv[a] > r, that's fine.
    #   Conflict if l <= inv[a] <= r, i.e., inv[a] >= l and inv[a] <= r.
    
    # So for fixed r, l is valid iff:
    #   For all a in [l, r]: inv[a] < l or inv[a] > r
    #   And a != inv[a] for all a in [l, r]
    
    # This is equivalent to:
    #   l > max{inv[a] : a in [l, r], inv[a] <= r}  -- but this depends on l, circular
    
    # Alternative: iterate l from r down to 1, and track if [l, r] is valid.
    # Start with l = r. Check if r is self-inverse. If yes, l = r-1.
    # Then for each new l added, check if inv[l] is in [l, r] or if l is in [l, r] where inv[l] <= r.
    # i.e., check if inv[l] >= l and inv[l] <= r. If so, conflict.
    
    # Two-pointer: move r from 1 to N, and for each r, find the maximum valid l.
    # As r increases, the valid l can only increase (or stay the same).
    
    l = 1
    for r in range(1, MOD):
        if time.time() > deadline:
            break
        
        # Ensure l <= r
        if l > r:
            l = r
        
        # Move l forward until [l, r] is valid
        while l <= r:
            # Check if adding l to the interval [l+1, r] (or just [l, r]) creates a conflict
            # We need to check if inv[l] is in [l, r]
            if inv[l] >= l and inv[l] <= r:
                # Conflict: l's inverse is in [l, r]
                l += 1
            else:
                # Also need to check: is there any x in [l, r] such that inv[x] == l?
                # That would mean x = inv[l], and we already checked inv[l] not in [l, r]
                # Wait, we need to check ALL x in [l, r], not just l.
                # But with two-pointer, when we increase r, we only need to check the new element r.
                # When we increase l, we only need to check the removed element l.
                
                # Actually, the two-pointer approach needs more care.
                # Let's use a different approach: for each r, binary search or linear scan for l.
                break
        
        # Now [l, r] should be valid (if l <= r)
        # But we need to verify: for all x in [l, r], inv[x] not in [l, r]
        # This is O(r-l) per r, which is O(N^2) total. For N=1008, that's ~1M ops, fine.
        
        # Let's just do a direct check for each r
        # Find max l such that [l, r] is valid
        valid_l = None
        for candidate_l in range(l, r + 1):
            # Check if [candidate_l, r] is valid
            valid = True
            for x in range(candidate_l, r + 1):
                ix = inv[x]
                if candidate_l <= ix <= r:
                    valid = False
                    break
            if valid:
                valid_l = candidate_l
                break
        
        if valid_l is not None:
            length = r - valid_l + 1
            if length > best_len:
                best_len = length
                best_start = valid_l
            l = valid_l  # For next r, l can only be >= current l
    
    return [best_start, best_len]
