"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # values are 1..1008
    
    # Compute modular inverse for each x in 1..1008
    # x * inv[x] ≡ 1 (mod 1009)
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Forbidden single values: 1 and 1008 (since 1*1%1009=1 and 1008*1008%1009=1)
    # So interval must not include 1 or 1008.
    # Therefore search in [2, 1007]
    
    lo_bound = 2
    hi_bound = 1007
    
    # We want the longest interval [s, e] such that for all x, y in [s, e], x*y % MOD != 1.
    # This is equivalent to: the set [s, e] does not contain any pair {x, inv[x]}.
    # Note: if x == inv[x], then x is a self-inverse. For prime p, x^2=1 => x=1 or x=p-1.
    # Both are excluded from our search range [2, 1007].
    # So all forbidden pairs are distinct pairs {x, y} with x != y.
    
    # Let's find all forbidden pairs within [2, 1007].
    # A pair {x, y} is forbidden if y == inv[x] and both x, y are in [2, 1007].
    # Since inv is an involution (inv[inv[x]] = x), we can just iterate x from 2 to 1007
    # and mark y = inv[x] as "connected" to x.
    
    # Actually, a simpler view:
    # An interval [s, e] is invalid if there exists x in [s, e] such that inv[x] is also in [s, e].
    # We want to maximize e - s + 1.
    
    # Strategy:
    # 1. Identify all "bad" points. A point x is "bad" relative to an interval if its inverse is also in the interval.
    # 2. Alternatively, we can scan all possible intervals? No, that's O(N^2).
    # 3. Better: For each starting point s, find the maximum e such that [s, e] is valid.
    #    As we extend e, the interval becomes invalid as soon as we include a number y such that inv[y] is already in [s, e-1].
    #    Wait, if we include y, we need to check if inv[y] is in [s, e].
    #    Since we are extending e, we only need to check if inv[y] >= s.
    #    If inv[y] < s, it's not in the interval, so it's fine.
    #    If inv[y] >= s, it IS in the interval (since inv[y] <= y? No, inv[y] can be larger or smaller).
    #    Actually, if we are adding y to the right end, the interval is [s, y].
    #    The new element is y. The only new pair formed involving y is {y, inv[y]}.
    #    This pair is invalid if inv[y] is in [s, y-1].
    #    So, for a fixed s, we can extend e as long as for all y in [s, e], inv[y] is NOT in [s, y-1].
    #    Wait, the condition is: for ALL x, y in [s, e], x*y != 1.
    #    This is equivalent to: for all x in [s, e], inv[x] is NOT in [s, e].
    #    (Note: if x=inv[x], it's a self-inverse, which we already excluded. So x != inv[x].)
    #    So, an interval [s, e] is valid iff for all x in [s, e], inv[x] < s OR inv[x] > e.
    
    # Let's refine the scan:
    # For each s in [2, 1007]:
    #   We want to find the largest e >= s such that for all x in [s, e], inv[x] is not in [s, e].
    #   Let's start with e = s.
    #   Check if inv[s] is in [s, s]. i.e., inv[s] == s. (False, as established).
    #   So [s, s] is always valid.
    #   Now try to extend e to s+1.
    #   The new element is y = s+1.
    #   We need to ensure that inv[y] is not in [s, y-1].
    #   Also, we need to ensure that for all existing x in [s, y-1], inv[x] is not y.
    #   But if inv[x] == y, then inv[y] == x. Since x is in [s, y-1], inv[y] is in [s, y-1].
    #   So the condition "for the new element y, inv[y] is not in [s, y-1]" is sufficient to guarantee that no new invalid pair is formed?
    #   Yes. If inv[y] is not in [s, y-1], then y is not the inverse of any existing element.
    #   And since y is new, the only pairs involving y are {y, inv[y]}.
    #   So the condition is: inv[y] < s OR inv[y] > y.
    #   Wait, inv[y] > y is always fine because the interval is [s, y], so any number > y is outside.
    #   So the only way adding y breaks the interval is if s <= inv[y] < y.
    #   i.e., inv[y] is in [s, y-1].
    
    # Algorithm:
    best_len = 0
    best_start = 2
    
    for s in range(lo_bound, hi_bound + 1):
        e = s
        # Extend e as long as possible
        while e < hi_bound:
            y = e + 1
            # Check if adding y breaks the interval [s, e]
            # It breaks if inv[y] is in [s, e]
            if inv[y] >= s and inv[y] <= e:
                break
            else:
                e += 1
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
        # Time check
        if time.time() > DEADLINE:
            break
            
    return [best_start, best_len]

# EVOLVE-BLOCK-END
