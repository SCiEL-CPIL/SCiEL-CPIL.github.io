"""Find the maximum interval in [1,1008] with no modular inverse pair (including self-inverse)."""
# EVOLVE-BLOCK-START
def construct():
    mod = 1009
    n = 1008  # domain [1, 1008]
    
    # Precompute modular inverses
    inv = [0] * (mod + 1)
    for x in range(1, mod):
        inv[x] = pow(x, mod - 2, mod)
    
    # A number x is "forbidden" if it participates in an inverse pair with some y
    # such that both x and y could be in the same interval.
    # More precisely: for an interval [a, b], we need that for all x, y in [a, b],
    # x*y % mod != 1.
    # This is equivalent to: for all x in [a, b], inv[x] is NOT in [a, b].
    # (Note: inv[x] is in [1, 1008] for all x in [1, 1008] since mod is prime)
    
    # So the condition is: for all x in [a, b], inv[x] < a or inv[x] > b.
    # Which means: a > max{inv[x] : x in [a,b], inv[x] < a} is not quite right.
    # Better: for all x in [a,b], inv[x] is not in [a,b].
    # Equivalently: min_{x in [a,b]} (if inv[x] in [a,b] then 1 else 0) == 0
    
    # Let's think of it as: [a,b] is valid iff for all x in [a,b], inv[x] < a or inv[x] > b.
    # This is equivalent to: max_{x in [a,b], inv[x] < a} inv[x] < a AND min_{x in [a,b], inv[x] > b} inv[x] > b
    # Wait, that's circular. Let me think differently.
    
    # [a,b] is valid iff there is NO x in [a,b] such that inv[x] is also in [a,b].
    # Since inv is an involution (inv[inv[x]] = x), this is symmetric.
    
    # For each x, the pair (x, inv[x]) is a "conflict" if x != inv[x] (i.e., x^2 != 1 mod p).
    # The self-inverse elements are x where x^2 ≡ 1 mod 1009, i.e., x ≡ 1 or x ≡ -1 ≡ 1008.
    
    # So forbidden configurations: if both x and inv[x] are in the interval, it's invalid.
    # Also, if x is self-inverse (x=1 or x=1008), then x*1 % 1009 = 1 only if... wait,
    # x * x % 1009 = 1 for self-inverse, so if x is in the interval, then x*x % 1009 = 1, 
    # which violates the condition. So 1 and 1008 can NEVER be in a valid interval.
    
    # So the domain for our interval is actually [2, 1007].
    
    # Now, for x in [2, 1007], inv[x] is also in [2, 1007] (since inv[x] != 1 and inv[x] != 1008 
    # would mean x != 1008 and x != 1, which is true).
    # Wait: inv[1] = 1, inv[1008] = 1008. For other x, inv[x] != x and inv[x] != 1, inv[x] != 1008.
    # Actually, could inv[x] = 1? That would mean x * 1 ≡ 1 mod 1009, so x ≡ 1. So no.
    # Could inv[x] = 1008? That would mean x * 1008 ≡ 1 mod 1009, so x ≡ 1008^{-1} ≡ 1008 (since 1008^2 = 1016064, and 1016064 mod 1009... let me check: 1008 = -1, so (-1)^2 = 1). So inv[1008] = 1008.
    # So for x in [2, 1007], inv[x] is in [2, 1007] and inv[x] != x.
    
    # So we need to find the longest interval [a, b] subset of [2, 1007] such that for all x in [a,b], inv[x] is NOT in [a,b].
    
    # This is equivalent to: the interval [a,b] does not contain both elements of any pair {x, inv[x]}.
    
    # Let's define a graph where vertices are 2..1007, and edges connect x to inv[x].
    # We need the longest contiguous subarray that contains no complete edge.
    
    # For each pair {x, inv[x]} with x < inv[x], the pair "blocks" any interval that contains both x and inv[x].
    # So the interval [a, b] is valid iff for every pair {x, inv[x]} with x < inv[x], NOT (a <= x and inv[x] <= b).
    
    # Equivalently: for every pair with x < inv[x], either x < a or inv[x] > b.
    
    # Let's precompute all pairs. For x in [2, 1007], if inv[x] > x, record the pair (x, inv[x]).
    
    pairs = []
    seen = set()
    for x in range(2, 1008):
        ix = inv[x]
        if ix > x and ix <= 1007:  # both in [2, 1007]
            if x not in seen:
                pairs.append((x, ix))
                seen.add(x)
                seen.add(ix)
    
    # Now, for a given interval [a, b], it's invalid if there exists a pair (x, y) with x < y such that a <= x and y <= b.
    # So [a, b] is valid iff for all pairs (x, y), NOT (a <= x and y <= b).
    # Which is equivalent to: for all pairs (x, y) with y <= b, we must have x < a.
    # I.e., a > max{x : (x, y) in pairs, y <= b}.
    
    # Alternatively: for a fixed a, what is the maximum b such that [a, b] is valid?
    # [a, b] is valid iff for all pairs (x, y) with x >= a, we have y > b.
    # So b < min{y : (x, y) in pairs, x >= a}.
    # Thus b_max(a) = min{y : (x, y) in pairs, x >= a} - 1, or 1007 if no such pair.
    
    # But we also need b >= a and b <= 1007.
    
    best_start = 2
    best_len = 0
    
    # Precompute for efficiency: for each a from 2 to 1007, compute b_max(a).
    # We can do this by iterating and maintaining the min y for pairs with x >= a.
    
    # Let's create an array min_y_for_x_ge[a] = min{y : (x, y) in pairs, x >= a}
    # Initialize to 1008 (larger than 1007)
    min_y = [1008] * (1009)  # indexed 0..1008
    for x, y in pairs:
        min_y[x] = min(min_y[x], y)
    
    # Now compute suffix minimum: suffix_min[a] = min(min_y[a], min_y[a+1], ..., min_y[1007])
    suffix_min = [1008] * (1009)
    suffix_min[1008] = 1008
    for a in range(1007, 0, -1):
        suffix_min[a] = min(min_y[a], suffix_min[a + 1])
    
    # Now for each a in [2, 1007]:
    # b_max = suffix_min[a] - 1
    # But we need b_max >= a for a valid interval starting at a.
    # Also b_max should be capped at 1007.
    
    for a in range(2, 1008):
        b_max = suffix_min[a] - 1
        if b_max > 1007:
            b_max = 1007
        if b_max >= a:
            length = b_max - a + 1
            if length > best_len:
                best_len = length
                best_start = a
    
    if best_len == 0:
        # Fallback: find any single element in [2, 1007]
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
