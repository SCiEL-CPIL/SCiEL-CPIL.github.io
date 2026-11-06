import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # forbidden[i] is True if i cannot be in the interval
    # Condition 1: Self-inverse (diagonal pairs)
    # x * x % MOD == 1 => x == 1 or x == MOD - 1 (since MOD is prime)
    # In range [1, 1008], these are 1 and 1008.
    # Condition 2: Distinct pairs
    # For any x in interval, if inv[x] is also in interval, it's invalid.
    
    # We can model this as: An interval [s, e] is valid if for all x in [s, e],
    # inv[x] is NOT in [s, e].
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # Precompute forbidden status for each index
    forbidden = [False] * (N + 1)
    for i in range(1, N + 1):
        if i == 1 or i == 1008:
            forbidden[i] = True
        # Note: if inv[i] is in the interval, it's a pair.
        # We don't mark inv[i] as "forbidden" globally, because inv[i] might be valid
        # if i is NOT in the interval. The constraint is on the *set* of numbers in the interval.
        
    # Two-pointer approach:
    # We want the longest subarray [s, e] such that no pair (x, y) with x*y % MOD == 1 exists within [s, e].
    # A pair exists if there is an x in [s, e] such that inv[x] is also in [s, e].
    # This is equivalent to: for all x in [s, e], inv[x] < s or inv[x] > e.
    
    best_len = 0
    best_start = 1
    
    s = 1
    # We can iterate e from 1 to N.
    # As e increases, we might violate the condition. We need to adjust s to fix it.
    # What causes a violation?
    # If we add e to the interval, we check if inv[e] is in [s, e-1].
    # If inv[e] is in [s, e-1], then the interval [s, e] is invalid.
    # To fix it, we must move s past inv[e]. So s = max(s, inv[e] + 1).
    # Also, we must ensure s itself is not forbidden (self-inverse).
    # Actually, if s is forbidden, the interval is invalid.
    # The self-inverse numbers are 1 and 1008.
    # If s=1, it's invalid. If e=1008, it's invalid.
    
    # Let's refine the two-pointer logic:
    # Maintain s such that [s, e] is valid.
    # Initially s=1.
    # For each e from 1 to N:
    #   1. If e is a self-inverse (1 or 1008), then any interval containing e is invalid.
    #      So we must move s to e+1. (Effectively skipping e).
    #      Wait, if e=1, s becomes 2. Interval [2,1] is empty.
    #      If e=1008, s becomes 1009.
    #   2. Check if inv[e] is in [s, e-1].
    #      If inv[e] >= s and inv[e] < e:
    #         Then the pair (inv[e], e) is in the interval.
    #         We must move s to inv[e] + 1 to exclude inv[e].
    #   3. After adjusting s, if s <= e, the interval [s, e] is a candidate.
    #      Check length.
    
    # Is it sufficient to only check inv[e]?
    # Yes, because any pair (x, y) with x < y would have been detected when y was added to the interval,
    # provided that x was still in the interval at that time.
    # If x was moved out by a previous step, then x is not in [s, e], so the pair doesn't exist.
    # If x is in [s, e-1], then when we add e, we detect the conflict and move s.
    
    # Let's trace:
    # s=1
    # e=1: self-inverse. s becomes 2. [2,1] empty.
    # e=2: inv[2] = 505. 505 >= 2? No. s remains 2. [2,2] valid. len=1.
    # ...
    # e=505: inv[505] = 2. 2 >= s (2)? Yes. 2 < 505? Yes.
    #         Conflict. s becomes 2+1=3.
    #         Interval [3, 505].
    #         Is it valid?
    #         We removed 2. Are there other pairs?
    #         Any pair (x, y) with y < 505 would have been handled when y was added.
    #         Since s was 2 until now, and we just moved it to 3, any x < 3 is gone.
    #         The only new pair involving 505 is (2, 505). We removed 2.
    #         So [3, 505] should be valid.
    
    s = 1
    for e in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Handle self-inverse
        if e == 1 or e == 1008:
            s = e + 1
            continue
            
        # Handle distinct pair
        ie = inv[e]
        if ie >= s and ie < e:
            s = ie + 1
            
        # Now [s, e] should be valid
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    # Edge case: if best_len is 0, return a trivial valid interval if possible?
    # The problem says "nonempty interval".
    # If no valid interval found (unlikely), return [1, 1] but 1 is invalid.
    # Let's ensure we return a valid one.
    # The loop should find at least length 1 for any e that isn't self-inverse and doesn't conflict with itself (impossible).
    # Wait, if e=2, s=2. [2,2] is valid. 2*2=4 != 1.
    # So best_len should be at least 1.
    
    if best_len == 0:
        # Fallback, though shouldn't happen
        return [2, 1]
        
    return [best_start, best_len]
