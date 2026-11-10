import time

def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element using a greedy algorithm.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 12
    chosen = []
    subset_sums = {0}
    forbidden = set()  # Set of differences s1 - s2 for s1, s2 in subset_sums, s1 != s2
    
    for k in range(n):
        if time.time() > deadline:
            break
            
        # Find the smallest positive integer not in chosen and not in forbidden
        # Note: forbidden contains differences s1 - s2. We need a_k such that 
        # (subset_sums + a_k) is disjoint from subset_sums.
        # This is equivalent to a_k not being in {s1 - s2 | s1, s2 in subset_sums, s1 != s2}.
        # Since a_k > 0, we only need to check positive differences.
        
        # Actually, it's simpler to just check if (subset_sums + candidate) intersects subset_sums.
        # But checking differences is faster if we maintain the forbidden set.
        
        # Let's maintain the forbidden set of positive differences.
        # When we add a new element a_k, the new subset sums are S_old and S_old + a_k.
        # The new differences will include:
        # 1. Old differences (already in forbidden)
        # 2. Differences within S_old (already in forbidden)
        # 3. Differences within S_old + a_k (same as differences within S_old)
        # 4. Differences between S_old and S_old + a_k: (s1 + a_k) - s2 = (s1 - s2) + a_k
        #    and s1 - (s2 + a_k) = (s1 - s2) - a_k
        
        # So we need to add all values (d + a_k) and (d - a_k) for d in forbidden?
        # No, the new forbidden set for the next step should contain all differences between any two elements in S_new.
        # S_new = S_old U (S_old + a_k)
        # Differences in S_new are:
        # - Differences in S_old (already known)
        # - Differences in S_old + a_k (same as differences in S_old)
        # - Differences between s1 in S_old and s2 + a_k in S_old + a_k: s1 - (s2 + a_k) = (s1 - s2) - a_k
        # - Differences between s1 + a_k in S_old + a_k and s2 in S_old: (s1 + a_k) - s2 = (s1 - s2) + a_k
        
        # So the new forbidden set is:
        # forbidden_new = forbidden U {d + a_k | d in forbidden} U {d - a_k | d in forbidden}
        # But we also need to consider differences where one is 0?
        # The differences are s1 - s2 for s1, s2 in S_new, s1 != s2.
        # If s1, s2 in S_old, d = s1 - s2 is in forbidden.
        # If s1, s2 in S_old + a_k, d = (s1' + a_k) - (s2' + a_k) = s1' - s2' is in forbidden.
        # If s1 in S_old, s2 in S_old + a_k, d = s1 - (s2' + a_k) = (s1 - s2') - a_k.
        #   Let d_old = s1 - s2'. This is in forbidden (or 0 if s1=s2').
        #   So d = d_old - a_k.
        # If s1 in S_old + a_k, s2 in S_old, d = (s1' + a_k) - s2 = (s1' - s2) + a_k.
        #   Let d_old = s1' - s2. This is in forbidden.
        #   So d = d_old + a_k.
        
        # So we can update forbidden efficiently.
        
        candidate = 1
        while True:
            if candidate in chosen:
                candidate += 1
                continue
            # Check if candidate is in forbidden
            # We need to check if (subset_sums + candidate) intersects subset_sums.
            # This is equivalent to candidate being in the set of differences {s1 - s2 | s1, s2 in subset_sums, s1 != s2}.
            # We maintain this set as 'forbidden'.
            
            if candidate in forbidden:
                candidate += 1
                continue
            
            # Also, we must ensure candidate is not equal to any previous element.
            # This is already checked by 'if candidate in chosen'.
            
            # If we get here, candidate is valid.
            chosen.append(candidate)
            
            # Update subset_sums
            new_sums = set()
            for s in subset_sums:
                new_sums.add(s + candidate)
            subset_sums.update(new_sums)
            
            # Update forbidden
            # New differences are of the form (d + candidate) and (d - candidate) for d in forbidden.
            # Also, we need to include candidate itself?
            # If s1 = s2 + candidate, then s1 - s2 = candidate.
            # Is candidate in forbidden? No, we just checked it's not.
            # But wait, the set of differences includes all s1 - s2 for s1, s2 in S_new.
            # The new differences are:
            # 1. Old differences (d)
            # 2. (d + candidate) for d in old differences
            # 3. (d - candidate) for d in old differences
            # 4. What about differences involving the new sums directly?
            #    The new sums are S_old and S_old + candidate.
            #    The differences between an element in S_old and an element in S_old + candidate are:
            #    s1 - (s2 + candidate) = (s1 - s2) - candidate
            #    (s1 + candidate) - s2 = (s1 - s2) + candidate
            #    where s1, s2 are in S_old.
            #    So the new differences are exactly {d + candidate, d - candidate | d in forbidden}.
            #    Plus the old differences.
            
            # So:
            new_forbidden = set(forbidden)
            for d in forbidden:
                new_forbidden.add(d + candidate)
                new_forbidden.add(d - candidate)
            
            # We only care about positive differences for the next step's check?
            # Actually, we need to check if a future candidate a_m is in the set of differences.
            # a_m will be positive. So we only need to store positive differences.
            # But d - candidate can be negative. We can filter out non-positive values.
            # However, if d - candidate is negative, it won't be a positive candidate.
            # But what if a future candidate is negative? No, we only pick positive integers.
            # So we can safely discard non-positive differences.
            
            forbidden = {d for d in new_forbidden if d > 0}
            
            break
            
        # Time check
        if time.time() > deadline:
            break
            
    # If we didn't reach n elements, return what we have (should not happen for n=12)
    if len(chosen) < n:
        # Fallback to powers of 2
        return [1 << j for j in range(n)]
        
    return chosen

# Test
if __name__ == "__main__":
    seq = construct()
    print("Sequence:", seq)
    print("Largest element:", max(seq))
    
    # Verify distinct subset sums
    n = len(seq)
    seen = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += seq[i]
        if s in seen:
            print("ERROR: Duplicate subset sum found!")
            break
        seen.add(s)
    else:
        print("All subset sums are distinct.")
