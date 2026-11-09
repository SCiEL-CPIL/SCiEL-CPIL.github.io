import time

def construct():
    """Find the longest interval [start, start+length-1] within [1, 1008] such that
    for every x, y in the interval (including x=y), x*y % 1009 != 1."""
    p = 1009
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Precompute modular inverses: inv[i] = i^(-1) mod p
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # For each starting position s, compute the maximum valid ending position e
    # such that [s, e] contains no inverse pair (including diagonal).
    # We use a suffix-minimum approach on the "blocking" inverse positions.
    
    # For each value v in [1, p-1], the interval containing v cannot also contain inv[v].
    # For a fixed start s, the interval [s, e] is valid iff for all v in [s, e],
    # inv[v] is not in [s, e]. This means inv[v] < s or inv[v] > e.
    # So e must be less than the minimum of {inv[v] : v in [s, e], inv[v] >= s}.
    
    # Efficient approach: for each s, the maximum valid e is the smallest value
    # among {v-1 : v in [s, p-1], inv[v] >= s} and {v-1 : v in [s, p-1], v in {1, p-1}}
    # that is >= s, minus 1. Actually, let's think differently.
    
    # For a fixed s, define bad[v] = True if v in [s, p-1] and (v == 1 or v == p-1 or (s <= inv[v] <= v)).
    # Wait, the constraint is: for all x, y in [s, e], x*y != 1 mod p.
    # This is equivalent to: for all x in [s, e], inv[x] is not in [s, e].
    # So for fixed s, the maximum valid e is the smallest value m >= s such that
    # either m is a self-inverse (m=1 or m=p-1) or inv[m] is in [s, m].
    # Then the valid interval is [s, m-1].
    
    # Let's precompute for each position v, the "constraint value" that limits e when s is fixed.
    # For a pair (a, b) with a*b = 1 mod p, a != b:
    #   If s <= a and s <= b, then e must be < max(a, b) if min(a,b) >= s.
    #   Actually, if s <= min(a,b), then e < max(a,b).
    #   If s > min(a,b) and s <= max(a,b), then the smaller one is already excluded, so no constraint.
    #   If s > max(a,b), no constraint.
    
    # So for each pair (a, b) with a < b and a*b = 1 mod p:
    #   For s in [1, a]: constraint is e < b
    #   For s in [a+1, b]: no constraint (a is excluded)
    #   For s > b: no constraint
    
    # Plus diagonal constraints: for v in {1, p-1}:
    #   For s <= v: constraint is e < v (i.e., e <= v-1)
    
    # Let's compute for each s, the minimum constraint value.
    # constraint[s] = min over all constraints active at s of the limit.
    # Then max_e[s] = constraint[s] - 1, and the interval [s, max_e[s]] is valid.
    
    # Initialize constraint to infinity for all s
    INF = p + 1
    constraint = [INF] * p
    
    # Diagonal constraints: v = 1 and v = p-1 = 1008
    for v in [1, p - 1]:
        # For s in [1, v], e must be < v, so constraint is v
        for s in range(1, v + 1):
            if v < constraint[s]:
                constraint[s] = v
    
    # Off-diagonal constraints: for each pair (a, b) with a < b, a*b = 1 mod p
    for a in range(1, p):
        b = inv[a]
        if a < b:
            # For s in [1, a], constraint is b
            for s in range(1, a + 1):
                if b < constraint[s]:
                    constraint[s] = b
    
    # Now compute the best interval
    best_len = 0
    best_start = 2
    
    for s in range(1, p):
        # Pruning
        if p - 1 - s + 1 <= best_len:
            break
        
        max_e = constraint[s] - 1
        if max_e >= s:
            current_len = max_e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
    
    return [best_start, best_len]
