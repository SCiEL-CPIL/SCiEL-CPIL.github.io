"""Find the maximum interval in [1,1008] with no modular inverse pair (including self-inverse)."""
# EVOLVE-BLOCK-START
def construct():
    mod = 1009
    n = 1008  # domain [1, 1008]
    
    # Precompute modular inverses using the efficient linear formula
    inv = [0] * (mod + 1)
    inv[1] = 1
    for i in range(2, mod):
        inv[i] = (mod - mod // i) * inv[mod % i] % mod
    
    # Self-inverse elements: x^2 ≡ 1 (mod 1009) => x ≡ 1 or x ≡ 1008
    # These can never be in a valid interval since x*x % 1009 == 1
    
    # Build pairs (x, inv[x]) where x < inv[x], both in [2, 1007]
    # For x in [2, 1007], inv[x] is also in [2, 1007] and inv[x] != x
    pairs = []
    seen = set()
    for x in range(2, 1008):
        ix = inv[x]
        if ix > x:
            pairs.append((x, ix))
            seen.add(x)
            seen.add(ix)
    
    # For each possible start a, find the maximum b such that [a, b] is valid.
    # [a, b] is valid iff for all pairs (x, y) with x >= a, we have y > b.
    # So b_max(a) = min{y : (x, y) in pairs, x >= a} - 1
    
    # Precompute suffix minimum of y-values for pairs with x >= a
    # min_y_at_x[x] = min{y : (x, y) in pairs} or infinity if no pair starts at x
    INF = 1008
    min_y_at_x = [INF] * (mod + 1)
    for x, y in pairs:
        min_y_at_x[x] = min(min_y_at_x[x], y)
    
    # Suffix minimum: suff[a] = min(min_y_at_x[a], min_y_at_x[a+1], ..., min_y_at_x[1007])
    suff = [INF] * (mod + 2)
    suff[1008] = INF
    for a in range(1007, 0, -1):
        suff[a] = min(min_y_at_x[a], suff[a + 1])
    
    best_start = 2
    best_len = 0
    
    for a in range(2, 1008):
        b_max = suff[a] - 1
        if b_max > 1007:
            b_max = 1007
        if b_max >= a:
            length = b_max - a + 1
            if length > best_len:
                best_len = length
                best_start = a
    
    if best_len == 0:
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
