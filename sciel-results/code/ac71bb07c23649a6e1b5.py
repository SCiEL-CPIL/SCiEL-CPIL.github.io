"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses for all elements in [1, 1008]
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # An interval [s, e] is valid if:
    # 1. For all x in [s, e], x*x % MOD != 1 (no self-inverse elements)
    #    Self-inverse elements are x where x*x ≡ 1 (mod MOD)
    #    Since MOD=1009 is prime, x*x ≡ 1 means x ≡ 1 or x ≡ -1 ≡ 1008
    # 2. For all x, y in [s, e], x*y % MOD != 1
    #    This means for any x in the interval, inv[x] must NOT be in the interval
    
    # Let's use a two-pointer approach to find the longest valid interval
    # For a fixed right endpoint r, find the smallest valid left endpoint l
    # such that [l, r] is valid.
    
    # An interval [l, r] is invalid if:
    # - It contains 1 (since 1*1=1)
    # - It contains 1008 (since 1008*1008 = 1008^2 mod 1009 = 1)
    # - For some x in [l, r], inv[x] is also in [l, r]
    
    # Key insight: if we fix the right endpoint r and expand left,
    # the interval becomes invalid when we include an element x such that
    # inv[x] is already in the interval.
    
    # Let's precompute for each position i, what is the "conflict" -
    # i.e., the smallest left endpoint that would conflict with i being in the interval.
    
    # For element i, its inverse is inv[i].
    # If i is in the interval [l, r], then inv[i] must NOT be in [l, r].
    # So if inv[i] < i, then l must be > inv[i] (i.e., l >= inv[i] + 1)
    # If inv[i] > i, then r must be < inv[i] (i.e., r <= inv[i] - 1)
    
    # Also, i cannot be 1 or 1008.
    
    # Let's think of it differently. For a candidate interval [l, r]:
    # It's valid iff for all x in [l, r]:
    #   - x != 1 and x != 1008
    #   - inv[x] < l or inv[x] > r
    
    # Using two pointers:
    # For each right endpoint r, maintain the minimum valid left endpoint l.
    # As r increases, l can only stay the same or increase.
    
    best_len = 0
    best_start = 1
    
    l = 1
    for r in range(1, N + 1):
        # Check if r itself is valid (not 1 or 1008)
        if r == 1 or r == 1008:
            l = r + 1
            continue
        
        # Now we need to ensure that for all x in [l, r], inv[x] is not in [l, r]
        # This is equivalent to: for all x in [l, r], inv[x] < l or inv[x] > r
        
        # The constraint that's hardest to satisfy as l decreases is:
        # For each x in [l, r], if inv[x] < x, we need l > inv[x]
        # If inv[x] > x, we need r < inv[x] (which is automatically satisfied if we're checking)
        
        # Actually, let me reconsider. For a fixed r, we need to find the smallest l
        # such that [l, r] is valid.
        
        # [l, r] is invalid if there exists x in [l, r] such that:
        #   x == 1 or x == 1008, or
        #   inv[x] is in [l, r]
        
        # For the second condition: inv[x] in [l, r] means l <= inv[x] <= r
        # So for each x in [l, r], we need NOT (l <= inv[x] <= r)
        # Which means for each x in [l, r]: inv[x] < l or inv[x] > r
        
        # Let's precompute for each r, what is the minimum l needed.
        # As we increase r, we add element r. The new constraints are:
        # - r must not be 1 or 1008 (handled above)
        # - For r itself: inv[r] must not be in [l, r], so if inv[r] < r, we need l > inv[r]
        #   i.e., l >= inv[r] + 1
        # - For existing elements x in [l_old, r-1]: we already had inv[x] < l_old or inv[x] > r-1
        #   Now with r, we need inv[x] < l or inv[x] > r
        #   If inv[x] > r-1, it might be that inv[x] == r, which would violate
        #   So if inv[x] == r, we need l > x, i.e., l >= x + 1
        
        # This is getting complex. Let me use a different approach.
        
        # For each r, compute the maximum "forbidden left" value.
        # forbidden_left = max over all x that would conflict if l <= some value
        
        # Actually, let's just check: for the current [l, r], is it valid?
        # If not, increment l until it is valid.
        
        # To check validity efficiently, note that [l, r] is invalid iff
        # there exists x in [l, r] such that x in {1, 1008} or inv[x] in [l, r]
        
        # Let's maintain a count or flag.
        
        # Simpler: just check if the interval is valid by checking the critical constraints.
        # For [l, r] to be valid, we need:
        # 1. l > 1 (so 1 is not in interval) -- actually we need 1 not in [l,r], so l > 1 or r < 1
        # 2. r < 1008 (so 1008 is not in interval)
        # 3. For all x in [l, r], inv[x] < l or inv[x] > r
        
        # Constraint 3 is the tricky one. Let's precompute for each possible l,
        # what is the maximum r such that [l, r] satisfies constraint 3.
        
        # Or: for each r, what is the minimum l such that [l, r] satisfies constraint 3?
        
        # Let's define: for a given r, min_valid_l(r) = max over all x in [1, r] of:
        #   if inv[x] <= r and inv[x] >= 1: we need l > inv[x] if inv[x] < x... 
        #   
        # Hmm, let me think more carefully.
        
        # For [l, r] to be valid, for every x in [l, r], we need inv[x] not in [l, r].
        # This is equivalent to: for every x in [l, r], either inv[x] < l or inv[x] > r.
        
        # Let's define conflict_r(x) = the condition that x conflicts with the interval ending at r.
        # x conflicts with [l, r] if x in [l, r] and inv[x] in [l, r].
        
        # For a fixed r, as l decreases, more elements x enter the interval, and for each
        # such x, we need inv[x] < l or inv[x] > r.
        
        # The most restrictive constraint comes from elements x where inv[x] is close to l.
        
        # Let me try a direct approach: for each r from 1 to N, find the smallest l
        # such that [l, r] is valid.
        
        # I'll maintain l and increase it as needed.
        
        # Check if current [l, r] is valid:
        # - 1 not in [l, r]: need l > 1 or r < 1
        # - 1008 not in [l, r]: need l > 1008 or r < 1008
        # - For all x in [l, r], inv[x] not in [l, r]
        
        # For the third condition, let's check if there's a violation.
        # A violation exists if there's x in [l, r] with l <= inv[x] <= r.
        
        # I can precompute for each position i, whether it's a "self-conflict" (1 or 1008)
        # and use the two-pointer method.
        
        # Let me restart with a cleaner two-pointer:
        pass
    
    # Let me redo this more carefully.
    l = 1
    best_len = 0
    best_start = 1
    
    for r in range(1, N + 1):
        # Ensure r is not a self-inverse element
        if r == 1 or r == 1008:
            l = r + 1
            continue
        
        # Now adjust l so that [l, r] is valid
        # [l, r] is invalid if:
        #   a) 1 in [l, r] -> l <= 1 <= r -> need l > 1 (since r >= 1)
        #   b) 1008 in [l, r] -> l <= 1008 <= r -> need l > 1008 or r < 1008
        #   c) exists x in [l, r] with inv[x] in [l, r]
        
        # Handle (a): if r >= 1, we need l > 1
        if l <= 1:
            l = 2
        
        # Handle (b): if r >= 1008, we need l > 1008
        if r >= 1008 and l <= 1008:
            l = 1009
        
        # Handle (c): we need to ensure no x in [l, r] has inv[x] in [l, r]
        # This is equivalent to: for all x in [l, r], inv[x] < l or inv[x] > r
        
        # Let's check if the current [l, r] violates (c).
        # If it does, we need to increase l.
        
        # The violation happens when there's x in [l, r] with l <= inv[x] <= r.
        # Note that if x is in [l, r] and inv[x] is in [l, r], then inv[inv[x]] = x,
        # so both x and inv[x] are in the interval.
        
        # To find the minimum l that avoids this, note that for a given r,
        # the constraint is: for all x in [l, r], inv[x] < l or inv[x] > r.
        
        # Let's define: for each x in [1, r], if inv[x] <= r, then we need l > inv[x] 
        # (if inv[x] < x) or we need l > x (if inv[x] > x and inv[x] <= r... wait, that doesn't make sense)
        
        # Actually: if x is in [l, r] and inv[x] <= r, then to avoid inv[x] in [l, r],
        # we need inv[x] < l, i.e., l > inv[x].
        
        # So for a fixed r, the constraint is: l > max{inv[x] : x in [1, r], inv[x] <= r, and x >= l}
        # But this is circular because l appears on both sides.
        
        # Let me think differently. For a fixed r, define:
        #   conflict_set = {x : x in [1, r] and inv[x] in [1, r]}
        # For [l, r] to be valid, we need [l, r] ∩ conflict_set = {} ... no, that's not right either.
        
        # OK let me just do it naively but efficiently.
        
        # For the current l and r, check if there's a conflict.
        # A conflict exists if there's x in [l, r] with inv[x] in [l, r].
        
        # I'll precompute for each i, inv[i]. Then for checking [l, r],
        # I can check: is there x in [l, r] such that l <= inv[x] <= r?
        
        # This is equivalent to: min over x in [l, r] of (whether inv[x] is in [l, r])
        
        # Let me use a different strategy: for each r, binary search or linearly find
        # the minimum valid l.
        
        # Actually, the two-pointer should work if I check properly.
        
        # Let me check if [l, r] has a conflict by checking the most recent elements.
        # When we added r, the new potential conflicts are:
        # - r itself: if inv[r] in [l, r-1] (since inv[r] != r unless r is self-inverse, which we excluded)
        #   Actually inv[r] could be in [l, r-1] or could be r (if self-inverse, excluded) or > r.
        # - Some x in [l, r-1] where inv[x] == r (i.e., x == inv[r])
        
        # So the new conflicts when adding r are:
        # 1. If inv[r] < r and inv[r] >= l: conflict (r and inv[r] both in interval)
        # 2. If inv[r] > r: no new conflict from r's perspective, but if inv[r] <= r... wait
        #    Actually if inv[r] > r, then r doesn't conflict with anything new.
        #    But we need to check: is there x in [l, r-1] with inv[x] == r?
        #    That would mean x == inv[r]. So if inv[r] >= l and inv[r] <= r-1, conflict.
        #    But inv[r] > r, so inv[r] > r-1, so no such x in [l, r-1].
        
        # Wait, I think I'm overcomplicating. Let me just verify:
        # When we extend the interval from [l, r-1] to [l, r], the new conflicts are:
        # - r with its inverse: if inv[r] in [l, r-1], conflict.
        # - Any x in [l, r-1] with inv[x] == r: this means x == inv[r]. 
        #   If inv[r] in [l, r-1], same as above.
        
        # So the only new conflict is: inv[r] in [l, r-1].
        # Which means: if l <= inv[r] <= r-1, we have a conflict.
        
        # Therefore, to resolve: if l <= inv[r] <= r-1, set l = inv[r] + 1.
        
        # But wait, we also need to make sure that after increasing l, the interval
        # [l, r] is still valid with respect to all previous elements. Since we only
        # increased l, and the previous interval [l_old, r-1] was valid, the new interval
        # [l_new, r] where l_new > l_old should still be valid for elements in [l_new, r-1]
        # (since they were valid before and we're just removing some elements from the left).
        # The only new element is r, and we handled its conflict.
        
        # So the algorithm is:
        # l = 1
        # for r = 1 to N:
        #   if r is self-inverse (1 or 1008): l = r + 1, continue
        #   if l <= inv[r] <= r-1: l = inv[r] + 1
        #   if l > r: continue (empty interval)
        #   update best
        
        # But we also need to handle the self-inverse check for all elements in [l, r],
        # not just r. However, since we skip r if it's self-inverse, and we set l > r
        # in that case, the interval [l, r] won't contain r. But what about other
        # self-inverse elements? The only self-inverse elements are 1 and 1008.
        # If 1 is in [l, r], that's a problem. But we skip r=1 by setting l=2.
        # For r > 1, if l <= 1, we need l > 1.
        # Similarly for 1008.
        
        # Let me add explicit checks.
        
        # Revised algorithm:
        l = 1
        best_len = 0
        best_start = 1
        
        for r in range(1, N + 1):
            # If r is self-inverse, skip it
            if r == 1 or r == 1008:
                l = r + 1
                continue
            
            # Ensure 1 is not in [l, r]
            if l <= 1 <= r:
                l = 2
            
            # Ensure 1008 is not in [l, r]
            if l <= 1008 <= r:
                l = 1009
            
            # Handle conflict with r's inverse
            if l <= inv[r] <= r - 1:
                l = inv[r] + 1
            
            # After adjustments, check if interval is non-empty
            if l > r:
                continue
            
            # The interval [l, r] should now be valid
            # (we've handled self-inverse and the new conflict from r)
            # But we should verify that no other conflicts exist.
            # Since we maintain the invariant that [l, r-1] was valid (or empty),
            # and we only added r and handled its conflict, [l, r] should be valid.
            
            current_len = r - l + 1
            if current_len > best_len:
                best_len = current_len
                best_start = l
        
        return [best_start, best_len]

# EVOLVE-BLOCK-END
