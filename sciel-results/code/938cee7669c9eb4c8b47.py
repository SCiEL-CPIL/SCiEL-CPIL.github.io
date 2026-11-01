<<<<<<< SEARCH
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # leave 10s margin for 120s limit
    
    if k != 5:
        # Fallback: use the original LUW construction for other k values
        return _luw_fallback(k, g)
    
    # For k=5, g=9: search for 5-regular graph with girth >= 9
    # Moore bound for (5,9): 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1+5+20+80+320 = 426
    # So n >= 426. Best known upper bound is 1116.
    
    # Try increasing n from Moore bound upward
    moore_bound = 1 + k + k*(k-1) + k*(k-1)**2 + k*(k-1)**3 + k*(k-1)**4
    # For k=5: 1 + 5 + 20 + 80 + 320 + 1280 = 1706? No, Moore bound for girth g:
    # For odd girth g = 2r+1: n >= 1 + k + k(k-1) + ... + k(k-1)^(r-1)
    # For g=9, r=4: n >= 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1+5+20+80+320 = 426
    
    moore = 1
    for i in range(4):  # r = 4 for girth 9
        moore += k * (k-1)**i
    
    # Search from moore bound upward
    # But random 5-regular graphs with girth 9 likely need n much larger than 426.
    # Known results suggest n in the hundreds to low thousands.
    
    # Strategy: try n values, for each try a few random graphs
    # Use time budget to guide search
    
    best_n = None
    best_adj = None
    
    # Try a range of n values
    # Start from a reasonable lower bound and go up
    # For practical purposes, try n from 426 to about 2000
    
    # To be efficient, try larger n first where random graphs are more likely
    # to have high girth (less likely to have short cycles)
    
    search_order = []
    # Try n values in a strategic order
    # Random regular graphs tend to have girth growing slowly with n
    # For 5-regular, expected girth is about 2*log_4(n)
    # For girth 9, need 2*log_4(n) >= 9, so log_4(n) >= 4.5, n >= 4^4.5 ≈ 1024
    
    # So try n around 1000-2000 first
    for n in range(1116, 1117):  # Start near known best
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            adj = _generate_random_5_reg(n)
            if adj is None:
                continue
            if _check_girth(adj, g, n):
                best_n = n
                best_adj = adj
                break
        if best_n is not None:
            break
    
    if best_n is None:
        # Try smaller n values (closer to Moore bound)
        for n in range(426, 1116, 10):
            if time.time() > deadline:
                break
            for attempt in range(30):
                if time.time() > deadline:
                    break
                adj = _generate_random_5_reg(n)
                if adj is None:
                    continue
                if _check_girth(adj, g, n):
                    best_n = n
                    best_adj = adj
                    break
            if best_n is not None:
                break
    
    if best_n is not None:
        # Convert adjacency list to edge list
        edges = []
        for u in range(best_n):
            for v in best_adj[u]:
                if u < v:
                    edges.append((u, v))
        return best_n, edges
    
    # Fallback: return the LUW construction
    return _luw_fallback(k, g)
=======
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # leave 20s margin for 150s limit
    
    if k != 5:
        # Fallback: use the original LUW construction for other k values
        return _luw_fallback(k, g)
    
    # For k=5, g=9: search for 5-regular graph with girth >= 9
    # Moore bound for (5,9): 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1+5+20+80+320 = 426
    # So n >= 426. Best known upper bound is 1116.
    
    # Strategy: try n values in a strategic order.
    # For 5-regular graphs, expected girth ~ 2*log_4(n)
    # For girth 9, need n >= 4^4.5 ≈ 1024
    # Try n values around and above this threshold
    
    best_n = None
    best_adj = None
    
    # Try n values in order of likelihood of success
    # Start with n values where random graphs are most likely to have girth >= 9
    # Then try smaller n values (closer to Moore bound)
    
    # First, try n values in the range where success is most likely
    # n = 1116 is the known best, but we might find something smaller
    # Try a range of n values
    
    # Search strategy: try multiple n values, each with a few attempts
    # Interleave to find the smallest n that works
    
    # Candidate n values to try, in order of preference
    # Start near the known best and work down, but also try some larger values
    candidate_ns = list(range(1116, 1117))  # Known best
    # Add smaller values to try
    for n in range(900, 1116, 5):
        candidate_ns.append(n)
    # Also try some values closer to the theoretical minimum
    for n in range(500, 900, 10):
        candidate_ns.append(n)
    
    # Try each candidate n with a few random attempts
    # Use time budget to limit total work
    attempts_per_n = 20
    
    for n in candidate_ns:
        if time.time() > deadline:
            break
        # Try a few random graphs for this n
        for attempt in range(attempts_per_n):
            if time.time() > deadline:
                break
            adj = _generate_random_5_reg(n)
            if adj is None:
                continue
            if _check_girth_nb(adj, g, n):
                best_n = n
                best_adj = adj
                break
        if best_n is not None:
            break
    
    if best_n is not None:
        # Convert adjacency list to edge list
        edges = []
        for u in range(best_n):
            for v in best_adj[u]:
                if u < v:
                    edges.append((u, v))
        return best_n, edges
    
    # Fallback: return the LUW construction
    return _luw_fallback(k, g)
>>>>>>> REPLACE
