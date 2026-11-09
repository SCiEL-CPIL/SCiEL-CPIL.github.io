import numpy as np
import time

def construct():
    t0 = time.monotonic()
    deadline = t0 + 60.0

    MOD = 1009

    # Compute modular inverses for all elements in [1, MOD-1]
    inv = np.array([0] * (MOD + 1), dtype=np.int32)
    for i in range(1, MOD):
        inv[i] = pow(i, -1, MOD)

    # Forbidden points: indices x in [1, 1008] that create a conflict.
    # Self-inverse elements (x * x % MOD == 1): x=1 and x=MOD-1=1008
    # For inverse pairs (a, b) where a < b and a*b % MOD == 1:
    #   An interval containing both a and b is invalid.
    #   To find the longest valid interval, we identify "barrier" points.
    #
    # Key insight: An interval [s, e] is valid iff for no x in [s,e], inv[x] is also in [s,e].
    # This means if we have a pair (a, b) with a < b, then no valid interval can contain both.
    # The "forbidden" structure: for each pair (a,b), any interval that spans from <=a to >=b is invalid.
    #
    # Equivalent formulation: define for each position x, the "constraint" that if x is in the interval,
    # then inv[x] must NOT be in the interval.
    #
    # For finding the longest valid contiguous interval:
    # A point x is a "barrier" in the sense that it splits the line.
    # Actually, the correct approach: for each inverse pair (a, b) with a < b,
    # the interval cannot contain both. This means the longest valid interval
    # is the maximum gap between consecutive "exclusion zones."
    #
    # Better: use the sliding window but implemented differently.
    # For each right endpoint r, find the minimum left endpoint l such that
    # [l, r] is valid. The constraint is: for all x in [l, r], inv[x] not in [l, r].
    # This is equivalent to: l > max{ inv[x] : x in [l, r] and inv[x] < l is false }
    # i.e., l must be > inv[x] for all x in [l, r] where inv[x] is in [l, r].
    #
    # Alternative: for each x, if x is in the interval, then inv[x] must be outside.
    # So for a candidate interval [l, r]: valid iff for all x in [l,r],
    #   inv[x] < l or inv[x] > r.
    #
    # Let's use a different approach: enumerate all pairs and find max gap.
    # For pair (a, b) with a < b: the interval cannot contain both.
    # This means the interval is either entirely in [1, a] U [b, 1008] (disconnected) 
    # or must avoid one of them.
    #
    # The cleanest O(n log n) approach:
    # For each x in [1, 1008], define the "conflict set." 
    # An interval [l, r] is valid iff no pair (x, inv[x]) is fully contained in [l, r].
    # 
    # Think of it as: each pair (min(a,b), max(a,b)) defines a "forbidden span."
    # The longest valid interval is the maximum length sub-interval that doesn't 
    # fully contain any forbidden span.
    #
    # For a pair (a, b) with a < b: it forbids any interval [l, r] with l <= a and r >= b.
    # So the valid intervals are those where for every pair, either l > a or r < b.
    #
    # Maximum gap approach: Consider all the "critical" points.
    # For each pair (a, b), the constraint is l > a OR r < b.
    # The longest interval is found by: for each possible left boundary l,
    # find the maximum r such that for all pairs (a,b) with a >= l, we have r < b.
    # i.e., r < min{ b : (a,b) is a pair and a >= l }
    #
    # Let's compute this directly.
    
    pairs = []
    seen_pairs = set()
    for x in range(1, MOD):
        y = inv[x]
        a, b = min(x, y), max(x, y)
        if a == b:
            # Self-inverse: this element cannot be in any valid interval
            # But since our interval is in [1, 1008] and we need x*y%1009!=1 for x=y,
            # self-inverse elements (1 and 1008) simply cannot be in the interval.
            pass
        elif (a, b) not in seen_pairs:
            seen_pairs.add((a, b))
            pairs.append((a, b))
    
    # Sort pairs by their first element a
    pairs.sort()
    
    # For each possible left boundary l (from 1 to 1008),
    # find the max valid right boundary r.
    # Constraint: for all pairs (a, b) where a >= l, we need r < b.
    # So r_max(l) = min{ b : (a,b) in pairs and a >= l } - 1
    # Also need to ensure l itself is not a self-inverse element.
    # And r must be >= l.
    
    # Also, self-inverse elements (1 and 1008) cannot be in the interval.
    # So l must not be 1 or 1008, and r must not be 1 or 1008.
    # Since we're looking for intervals in [1, 1008], and 1 is the smallest,
    # if l=1, that's invalid. So effectively l >= 2.
    # Similarly r <= 1007.
    
    # Precompute: for each l, what is the min b among pairs with a >= l?
    # Use suffix minimum of b values.
    
    # Map: for each a value, the minimum b among pairs starting at a or later.
    # Since pairs are sorted by a, we can compute suffix min of b.
    
    max_a = 1008
    min_b_suffix = [max_a + 1] * (max_a + 2)
    
    # For each a from max_a down to 1, min_b_suffix[a] = min(b for pairs with a' >= a)
    # First, for each a, store the min b for pairs exactly at a
    min_b_at_a = [max_a + 1] * (max_a + 2)
    for a, b in pairs:
        if b < min_b_at_a[a]:
            min_b_at_a[a] = b
    
    for a in range(max_a, 0, -1):
        min_b_suffix[a] = min(min_b_at_a[a], min_b_suffix[a + 1])
    
    best_len = 0
    best_start = 2
    
    for l in range(2, 1008):
        # l must not be self-inverse (1 and 1008 are the only ones, so l in [2,1007] is fine)
        # r_max = min_b_suffix[l] - 1, but also r must be <= 1007
        r_max = min(min_b_suffix[l] - 1, 1007)
        if r_max >= l:
            length = r_max - l + 1
            if length > best_len:
                best_len = length
                best_start = l
        
        if time.monotonic() > deadline - 5.0:
            break
    
    # Edge case: if no valid interval found (shouldn't happen), return minimal
    if best_len == 0:
        # Find any valid single element
        for x in range(2, 1008):
            if inv[x] < 1 or inv[x] > 1008 or inv[x] == x:
                # Actually inv[x] == x means self-inverse, which is invalid
                # inv[x] < 1 or inv[x] > 1008 means inv is outside [1,1008]
                # But inv is always in [1, 1008] for x in [1, 1008]
                # So we need inv[x] not in {x} and the interval [x,x] means only x is present
                # x*x % 1009 != 1 is required
                if x * x % MOD != 1:
                    best_len = 1
                    best_start = x
                    break
    
    return [best_start, best_len]
