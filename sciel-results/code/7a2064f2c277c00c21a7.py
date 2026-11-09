import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    p = 1009
    max_val = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Self-inverse elements: x where x^2 ≡ 1 (mod p), i.e., x = 1 or x = p-1
    # These cannot be in any valid interval (since x*x % p == 1)
    
    # For each x in [2, p-2], compute its inverse pair partner
    # A valid interval [s, e] must not contain both elements of any inverse pair
    # Also must not contain 1 or p-1
    
    # Strategy: model as finding the longest contiguous interval in [2, p-2]
    # such that no two elements in the interval are modular inverses of each other.
    # 
    # Key insight: For each pair (a, b) with a < b and a*b ≡ 1 (mod p),
    # at most one of a, b can be in the interval.
    # 
    # This is equivalent to: the interval [s, e] is valid iff
    # for all x in [s, e], inv[x] is NOT in [s, e].
    # 
    # We can use a sweep-line / two-pointer approach:
    # For a fixed left endpoint s, we want to find the maximum right endpoint e
    # such that [s, e] is valid.
    # 
    # As we extend e from s to p-2, we need to check that for each new element e,
    # inv[e] is not in [s, e]. Since inv[e] could be < s (fine) or > e (fine) or in [s, e] (bad).
    # 
    # More efficiently: for each position i, define "blocker[i]" = the other element in its inverse pair.
    # For i in [2, p-2], blocker[i] = inv[i] (which is also in [2, p-2] for non-self-inverse elements).
    # Self-inverse elements in [2, p-2]: none (only 1 and p-1 are self-inverse mod 1009).
    # 
    # So for each i in [2, p-2], we have a pair {i, inv[i]} where i != inv[i].
    # 
    # Two-pointer: maintain window [s, e]. The window is valid iff no pair has both endpoints in [s, e].
    # We can track: for the current window, which pairs have both endpoints inside?
    # 
    # Alternative: for each right endpoint e, compute the minimum valid start s.
    # s must be > max{ a : a < e, a * e ≡ 1 (mod p), a >= s } -- circular.
    # 
    # Better: precompute for each e in [2, p-2], the set of "conflicts" that would invalidate
    # an interval ending at e. Specifically, if inv[e] < e, then any interval [s, e] with s <= inv[e]
    # is invalid. So the constraint from e is: s > inv[e] (if inv[e] < e and inv[e] >= 2).
    # 
    # But also, for any x in [s, e-1] that was already in the interval, we need inv[x] not in [s, e].
    # This means: if inv[x] == e, then x must not be in [s, e], i.e., s > x.
    # 
    # So for a fixed e, the constraints on s are:
    # 1. s >= 2
    # 2. For x = e: if inv[e] < e and inv[e] >= 2, then s > inv[e]
    # 3. For each x in [2, e-1] with inv[x] == e: s > x
    #    (but inv[x] == e means x = inv[e], so this is the same as constraint 2 when inv[e] < e)
    # 
    # Wait, that's not quite right. The issue is that for ALL x in [s, e], inv[x] must be outside [s, e].
    # This includes pairs where both elements are < e.
    # 
    # Let me think differently. Define for each position i in [2, p-2]:
    #   partner[i] = inv[i]
    # 
    # An interval [s, e] is valid iff for all i in [s, e], partner[i] < s or partner[i] > e.
    # 
    # This is equivalent to: there is no i in [s, e] such that s <= partner[i] <= e.
    # 
    # For a two-pointer approach:
    # - Start with s = 2, e = 2
    # - Expand e while the window remains valid
    # - When adding e+1 would make the window invalid, increment s until valid again
    # 
    # To check validity efficiently when expanding e:
    # - When we add a new element e to the window, we need to check if partner[e] is in [s, e].
    #   If partner[e] < s or partner[e] > e, it's fine.
    #   If s <= partner[e] <= e, then the window is invalid.
    # 
    # When we shrink from the left (increment s), we remove element s.
    # 
    # But we also need to check: for all existing elements x in [s, e], is partner[x] in [s, e]?
    # This is hard to maintain incrementally without a data structure.
    # 
    # Alternative: for each e, compute the maximum valid s.
    # For a fixed e, s must satisfy:
    #   For all x in [s, e]: partner[x] < s or partner[x] > e
    #   
    # This means: s > max{ partner[x] : x in [s, e], partner[x] < x } 
    # AND e < min{ partner[x] : x in [s, e], partner[x] > x }
    # 
    # The second condition depends on s (through the range [s, e]), making it circular.
    # 
    # Let's use a different formulation. For each e from 2 to p-2:
    #   Let S_e = { x in [2, e] : partner[x] > e }
    #   Let T_e = { x in [2, e] : partner[x] < x }  (partner is to the left of x)
    #   
    # For [s, e] to be valid:
    #   - No x in [s, e] has partner[x] in [s, e]
    #   - Equivalently: for all x in [s, e] with partner[x] < x: partner[x] < s
    #     AND for all x in [s, e] with partner[x] > x: partner[x] > e
    #   
    # The second condition: for all x in [s, e] with partner[x] > x, we need partner[x] > e.
    # This is automatically satisfied if we only consider x where partner[x] > e.
    # But if there's an x in [s, e] with x < partner[x] <= e, that's a conflict.
    # 
    # So for fixed e, define:
    #   conflict_left[e] = max{ x : x in [2, e], partner[x] > x, partner[x] <= e }
    #   If conflict_left[e] exists, then s must be > conflict_left[e] to avoid including x
    #   (since if s <= x and x <= e, and partner[x] in (x, e], then both x and partner[x] are in [s,e])
    #   
    #   Also, for x in [s, e] with partner[x] < x: we need partner[x] < s, i.e., s > partner[x].
    #   So s > max{ partner[x] : x in [s, e], partner[x] < x }
    #   
    # This is still circular. Let's just do O(n^2) since n ~ 1006.
    
    best_len = 0
    best_start = 2
    
    # For each start s in [2, p-2], find max valid e
    for s in range(2, p - 1):
        # Early termination
        max_possible_len = (p - 2) - s + 1
        if max_possible_len <= best_len:
            break
        
        # Check time
        if time.time() > deadline:
            break
        
        # Find max e such that [s, e] is valid
        valid_e = s - 1  # start with no valid interval
        for e in range(s, p - 1):
            # Check if [s, e] is valid: for all x in [s, e], inv[x] not in [s, e]
            # We can check incrementally: when adding e, check if inv[e] is in [s, e]
            # But we also need to ensure all previous elements are still fine
            # Since we're extending e, the only new potential conflict is from element e
            # and from element inv[e] if it's in [s, e-1]
            
            # Check: is inv[e] in [s, e]?
            ie = inv[e]
            if s <= ie <= e:
                # Conflict: both e and inv[e] are in [s, e]
                # So [s, e] is invalid, and any larger e will also be invalid
                # (since the conflict remains)
                break
            
            # Also need to check: for all x in [s, e-1], inv[x] not in [s, e]
            # But we already checked this when we extended to e-1.
            # The only new issue is: could inv[x] == e for some x in [s, e-1]?
            # That would mean x = inv[e], which we already checked (ie = inv[e]).
            # If ie < s, then x = ie is not in [s, e-1], so no new conflict.
            # If ie > e, then x = ie is not in [s, e-1], so no new conflict.
            # If s <= ie <= e-1, then we'd have caught it when we added ie.
            
            # Wait, this isn't quite right. When we extend from e-1 to e,
            # we need to check that no x in [s, e-1] has inv[x] == e.
            # inv[x] == e iff x == inv[e].
            # So we just need to check if inv[e] is in [s, e-1].
            # If inv[e] is in [s, e-1], then both inv[e] and e are in [s, e], conflict.
            # If inv[e] == e, that's a self-inverse, but we've excluded those.
            
            # So the check is: is inv[e] in [s, e]?
            # If yes, break. If no, [s, e] is valid (assuming [s, e-1] was valid).
            
            valid_e = e
        
        length = valid_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
