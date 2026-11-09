# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # values are 1..1008
    
    # Compute modular inverse for each x in 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse: x where inv[x] == x
    # For prime p, x^2 = 1 mod p => x = 1 or x = p-1
    # So 1 and 1008 are self-inverse, must be excluded
    
    # Search space: [2, 1007]
    lo_bound = 2
    hi_bound = 1007
    
    # For each x in [2, 1007] where x != inv[x], we have a forbidden pair {x, inv[x]}
    # We need the longest interval [s, e] in [2, 1007] that doesn't contain both elements of any such pair
    
    # Two-pointer approach:
    # For a fixed left endpoint s, the right endpoint e is limited by:
    #   For every pair (a, b) with a < b where a >= s, we need e < b
    #   So e < min(b for all pairs (a,b) with a >= s and a < b)
    #   Also e < min(b for all pairs (a,b) with b >= s and a < b) -- but if a < s, then a is not in interval, so no constraint
    #   Wait, let me reconsider.
    
    # A pair {a, b} (a < b) is "fully contained" in [s, e] iff s <= a and b <= e
    # So for a given s, we need: for all pairs (a, b) with a >= s, we must have e < b
    # Therefore: e_max(s) = min(b for all pairs (a,b) with a >= s) - 1, but also e <= hi_bound
    # And we also need e >= s (nonempty interval)
    
    # Let's collect all pairs (min(x,inv[x]), max(x,inv[x])) for x in [2,1007] where x != inv[x]
    pairs = []
    seen = set()
    for x in range(lo_bound, hi_bound + 1):
        ix = inv[x]
        if ix == x:
            continue  # self-inverse, already excluded by bounds
        a = min(x, ix)
        b = max(x, ix)
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    
    # Sort pairs by a (the smaller element)
    pairs.sort(key=lambda p: p[0])
    
    # For each possible start s in [2, 1007], find max e
    # e must be < b for every pair (a, b) with a >= s
    # Precompute: for each s, what is the minimum b among pairs with a >= s?
    
    # Let's build an array min_b_from[s] = min(b for pairs (a,b) with a >= s)
    # If no such pair, min_b_from[s] = infinity (meaning no constraint from pairs starting at or after s)
    
    max_a = hi_bound
    # Initialize with a large value
    INF = hi_bound + 1
    min_b_from = [INF] * (hi_bound + 2)
    
    # Process pairs in reverse order of a
    # For each a value, min_b_from[a] = min(b for pairs with that a, and min_b_from[a+1])
    # Actually, min_b_from[s] = min over all pairs (a,b) with a >= s of b
    
    # Let's do it by iterating s from hi_bound down to lo_bound
    # Keep track of the minimum b seen so far for pairs with a >= current s
    
    # Better: create an array where for each a, we store the min b for pairs starting at a
    min_b_at_a = [INF] * (hi_bound + 2)
    for (a, b) in pairs:
        if b < min_b_at_a[a]:
            min_b_at_a[a] = b
    
    # Now min_b_from[s] = min(min_b_at_a[s], min_b_at_a[s+1], ..., min_b_at_a[hi_bound])
    # Compute suffix minimum
    min_b_from[hi_bound + 1] = INF
    for s in range(hi_bound, lo_bound - 1, -1):
        min_b_from[s] = min(min_b_at_a[s], min_b_from[s + 1])
    
    # Now for each s, the maximum e is min(min_b_from[s] - 1, hi_bound)
    # And we need e >= s for a nonempty interval
    best_start = lo_bound
    best_len = 0
    
    for s in range(lo_bound, hi_bound + 1):
        max_e = min(min_b_from[s] - 1, hi_bound)
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Should not happen, but safety
        pass
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
