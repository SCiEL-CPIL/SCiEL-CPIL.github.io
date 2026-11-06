"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    MOD = 1009
    N = 1008  # range is [1, 1008]
    
    # Compute modular inverse for each x in [1, 1008]
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Forbidden self-inverse: x where x * x % MOD == 1, i.e., x=1 or x=1008
    # These cannot appear in any valid interval
    
    # For each position i (1-indexed), determine the "conflict" - 
    # if i and inv[i] are both in [1, N], they form a conflict edge.
    # A valid interval must not contain both endpoints of any conflict edge,
    # and must not contain self-inverse elements (1, 1008).
    
    # Strategy: For each possible end position e, find the maximum start s
    # such that [s, e] is valid. Use a backward pass maintaining the constraint.
    
    # For a conflict edge (a, b) where a != b and a, b in [1, N]:
    #   An interval [s, e] is invalid if s <= min(a,b) and e >= max(a,b)
    #   i.e., the interval contains both a and b.
    # For self-inverse (a == b, i.e., a=1 or a=1008):
    #   An interval [s, e] is invalid if s <= a <= e.
    
    # For each end e, the start s must satisfy:
    #   s > max over all conflict edges (a,b) with max(a,b) <= e of min(a,b)
    #   s > max over all self-inverse a with a <= e of a
    #   Also s <= e and s >= 1
    
    # Let's precompute for each position the "bound" it imposes.
    # For a conflict edge (a, b) with a < b:
    #   For any end e >= b, we need s > a, i.e., s >= a+1
    #   So the bound at end e is at least a+1
    # For self-inverse at position p (p=1 or p=1008):
    #   For any end e >= p, we need s > p, i.e., s >= p+1
    
    # So for each end e, min_start[e] = max(1, max of all bounds from edges/self-inverse with trigger <= e)
    
    # Let's build an array where for each "trigger" position t (which is max(a,b) for edge or p for self-inverse),
    # we record the required lower bound on start.
    
    # bounds_at_trigger[t] = max required start lower bound when end >= t
    bounds = [1] * (N + 1)  # 1-indexed, default bound is 1 (start >= 1)
    
    # Process self-inverse elements
    for x in range(1, N + 1):
        if x * x % MOD == 1:
            # x is self-inverse, cannot be in interval
            # For any end e >= x, start must be > x
            if x <= N:
                bounds[x] = max(bounds[x], x + 1)
    
    # Process conflict edges (x, inv[x]) where x != inv[x]
    processed = set()
    for x in range(1, N + 1):
        y = inv[x]
        if y == x:
            continue  # already handled as self-inverse
        if y < 1 or y > N:
            continue
        # Edge between x and y
        a, b = min(x, y), max(x, y)
        if a == b:
            continue
        # For any end e >= b, start must be > a
        trigger = b
        bound_val = a + 1
        if bound_val > bounds[trigger]:
            bounds[trigger] = bound_val
    
    # Now do a backward/forward pass: for each end e, the effective minimum start is
    # max(bounds[1], bounds[2], ..., bounds[e]) — but actually we need:
    # min_start[e] = max(1, max{bounds[t] : t <= e})
    # Wait, no. The constraint is: for each edge with trigger t, if e >= t then s >= bounds[t].
    # So min_start[e] = max(1, max{bounds[t] : t <= e})
    
    # Actually that's a prefix max. Let me compute it.
    min_start = [1] * (N + 1)
    running_max = 1
    for e in range(1, N + 1):
        running_max = max(running_max, bounds[e])
        min_start[e] = running_max
    
    # Now for each end e, the valid starts are [min_start[e], e].
    # The length of the longest valid interval ending at e is e - min_start[e] + 1 (if min_start[e] <= e)
    
    best_start = 1
    best_length = 0
    for e in range(1, N + 1):
        s = min_start[e]
        if s <= e:
            length = e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
