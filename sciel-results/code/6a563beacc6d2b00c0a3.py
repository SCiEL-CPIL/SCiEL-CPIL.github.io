import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # For each position R (as right endpoint of interval), we need the minimum
    # left endpoint L such that [L, R] contains no complete conflict edge.
    #
    # Conflict edges: for each x in [1, N], edge between x and inv[x].
    # If x == inv[x] (self-loop at 1 and 1008), the edge is a self-loop.
    # An interval [L, R] is valid iff for every edge (a, b) with a <= b,
    # it is NOT the case that L <= a and b <= R.
    #
    # For a fixed R, the constraint from edge (a, b) where b <= R is:
    #   L > a  (i.e., L >= a + 1)
    # So the minimum valid L for endpoint R is:
    #   L_min(R) = max over all edges (a, b) with b <= R of (a + 1)
    #            = max(1, max_{edges (a,b), b<=R} (a+1))
    #
    # We can compute this in a single forward pass:
    #   For each R from 1 to N, update the running max bound.
    #   When we reach position R, edges where the larger endpoint equals R
    #   become active. For edge (a, b) with a <= b, the larger endpoint is b,
    #   so edge (a, b) activates when R reaches b.
    #   The constraint it imposes is L >= a + 1.
    #
    # For self-loops (x = inv[x]), the edge is (x, x), activating at R = x,
    # imposing L >= x + 1.

    # Build list of edges grouped by their larger endpoint
    # For each x, the edge is (min(x, inv[x]), max(x, inv[x]))
    # But we only need to process each edge once. Since inv is an involution,
    # edge (x, inv[x]) = edge (inv[x], x). Process each edge once.
    
    # max_conflict_bound[R] = maximum of (a+1) over all edges (a,b) with b == R
    # Then L_min(R) = max(L_min(R-1), max_conflict_bound[R])
    
    max_conflict_bound = [0] * (N + 2)  # indexed 1..N
    
    for x in range(1, N + 1):
        a = x
        b = inv[x]
        lo = min(a, b)
        hi = max(a, b)
        # Edge (lo, hi) activates when R reaches hi
        # It requires L >= lo + 1
        if lo + 1 > max_conflict_bound[hi]:
            max_conflict_bound[hi] = lo + 1
    
    # Forward pass: compute L_min(R) for each R
    best_length = 0
    best_start = 1
    running_max_bound = 1  # L must be at least 1
    
    for R in range(1, N + 1):
        if max_conflict_bound[R] > running_max_bound:
            running_max_bound = max_conflict_bound[R]
        
        L_min = running_max_bound
        # L_min must also be <= R for a nonempty interval
        if L_min <= R:
            current_length = R - L_min + 1
            if current_length > best_length:
                best_length = current_length
                best_start = L_min
        else:
            # No valid interval ending at R (L_min > R means even single element R conflicts)
            # This happens when R is self-inverse: L_min = R+1 > R
            # In that case, skip R as an endpoint
            pass
        # Time check
        if time.time() > deadline:
            break

    if best_length == 0:
        # Fallback: any single element that is not self-inverse
        for x in range(1, N + 1):
            if x != 1 and x != 1008:
                best_start = x
                best_length = 1
                break
        if best_length == 0:
            best_start = 1
            best_length = 1

    return [best_start, best_length]
