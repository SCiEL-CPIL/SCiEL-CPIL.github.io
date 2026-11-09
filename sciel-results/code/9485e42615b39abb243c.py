def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MOD = 1009
    N = 1008  # values are 1..1008

    # Compute modular inverse for each x in 1..1008
    # x * inv[x] ≡ 1 (mod 1009)
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        # Fermat's little theorem: x^(MOD-2) mod MOD for prime MOD
        inv[x] = pow(x, MOD - 2, MOD)

    # The problem: find longest contiguous interval [s, e] in [1, 1008]
    # such that for ALL x, y in [s, e], x * y % MOD != 1.
    # This is equivalent to: for all x in [s, e], inv[x] is NOT in [s, e].
    # (Because if inv[x] is in [s,e], then x * inv[x] % MOD == 1.)
    # Self-inverse elements (x = inv[x]) mean x^2 ≡ 1 mod 1009, i.e., x=1 or x=1008.
    # These can never be in any valid interval.

    # Strategy from the inspiration:
    # For each x, define the "conflict" as the pair {x, inv[x]}.
    # An interval [s, e] is valid iff it does NOT contain both members of any conflict pair.
    # 
    # Key insight: For a fixed start s, as we extend e, the interval becomes invalid
    # at the first e where some conflict pair is fully contained in [s, e].
    # 
    # More efficiently: For each position i, consider the constraint it imposes.
    # If i is in the interval, then inv[i] must NOT be in the interval.
    # So for interval [s, e]: for all i in [s, e], we need inv[i] < s or inv[i] > e.
    #
    # Let's use the "blocker" idea from the strategy:
    # For each i, if i < inv[i], then including i forces us to exclude inv[i].
    # The "small" endpoint of the pair {i, inv[i]} is min(i, inv[i]).
    # 
    # Actually, let me think about this more carefully.
    # For interval [s, e] to be valid: no pair {x, inv[x]} is fully contained in [s, e].
    # 
    # For each pair {a, b} where a = min(x, inv[x]), b = max(x, inv[x]),
    # the interval [s, e] is invalid if s <= a and b <= e.
    # 
    # So for a fixed start s, the maximum valid e is:
    # e_max(s) = min over all pairs {a,b} with a >= s of (b - 1), but only if b > s.
    # Wait, that's not quite right either.
    # 
    # Let me think again. For fixed s, we want the largest e such that
    # for all pairs {a, b} (where a = min(x,inv[x]), b = max(x,inv[x])),
    # it is NOT the case that s <= a AND b <= e.
    # 
    # Equivalently: for all pairs with a >= s, we need b > e.
    # So e < min{b : a >= s, a < b}
    # But we also need e >= s, and we need to make sure the interval is non-empty.
    #
    # Actually, we need: for all pairs {a, b} with a >= s, b > e.
    # This means e < min{b : a >= s}.
    # But we also need e >= s.
    # And we need to exclude self-inverse pairs (where a = b), but those are just x=1,1008.
    
    # Let's precompute for each possible start s, the minimum b among all pairs with a >= s.
    # Then e_max(s) = min_b(s) - 1, where min_b(s) = min{b : a >= s}.
    # But we need e_max(s) >= s for the interval to be non-empty.
    
    # Build list of (a, b) pairs where a < b (excluding self-inverse)
    pairs = []
    seen = set()
    for x in range(1, MOD):
        ix = inv[x]
        if x == ix:
            continue  # self-inverse, skip
        a = min(x, ix)
        b = max(x, ix)
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    
    # For each possible start s from 1 to 1008, compute min_b(s) = min{b : a >= s}
    # We can do this by sorting pairs by a and using a suffix minimum.
    
    # Sort pairs by a
    pairs.sort(key=lambda p: p[0])
    
    # For efficiency, let's create an array min_b_from[s] for s from 1 to 1008
    # min_b_from[s] = min{b : (a,b) in pairs, a >= s}
    # If no such pair exists, min_b_from[s] = infinity
    
    INF = float('inf')
    min_b_from = [INF] * (MOD + 2)
    
    # Process pairs in order of decreasing a
    # For each s, min_b_from[s] = min(min_b_from[s+1], min{b : a == s})
    
    # First, for each a value, find the minimum b
    min_b_at_a = [INF] * (MOD + 1)
    for a, b in pairs:
        if b < min_b_at_a[a]:
            min_b_at_a[a] = b
    
    # Suffix minimum
    for s in range(MOD, 0, -1):
        min_b_from[s] = min(min_b_at_a[s], min_b_from[s + 1])
    
    # Now for each start s, the maximum valid end is:
    # e_max(s) = min_b_from[s] - 1
    # But we need e_max(s) >= s for a non-empty interval.
    # Also, s itself must not be self-inverse.
    
    best_start = 2
    best_len = 1  # At least a single element (not 1 or 1008) is valid
    
    for s in range(1, MOD):
        # Check if s is self-inverse
        if inv[s] == s:
            continue
        
        e_max = min_b_from[s] - 1
        if e_max < s:
            continue
        
        # The interval [s, e_max] should be valid. Let's verify.
        # But first, check length
        length = e_max - s + 1
        if length > best_len:
            best_len = length
            best_start = s
        
        # Deadline check
        if time.time() - start_time > DEADLINE:
            break
    
    # Verify the result
    final_start = best_start
    final_end = best_start + best_len - 1
    valid = True
    for x in range(final_start, final_end + 1):
        if final_start <= inv[x] <= final_end:
            valid = False
            break
    
    if not valid:
        # Fallback: do a brute-force search for a valid interval
        # Find any valid single element
        for s in range(2, MOD):
            if inv[s] != s:
                return [s, 1]
        return [2, 1]
    
    return [best_start, best_len]
