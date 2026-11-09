<<<<<<< SEARCH
def _try_solve_multi(adj, branches, deadline):
    """Try multiple orderings of pairs to find a valid path assignment."""
    bs = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    
    # Try natural order
    r = _try_solve(adj, branches, pairs)
    if r is not None:
        return r
    
    # Try sorted by shortest path
    order = _order_pairs_by_shortest(adj, branches)
    r = _try_solve(adj, branches, order)
    if r is not None:
        return r
    
    # Try reverse shortest
    r = _try_solve(adj, branches, list(reversed(order)))
    if r is not None:
        return r
    
    # Try random orders (limited by time)
    for _ in range(10):
        if time.time() > deadline:
            break
        random.shuffle(pairs)
        r = _try_solve(adj, branches, pairs)
        if r is not None:
            return r
    
    return None
=======
def _try_solve_multi(adj, branches, deadline):
    """Try multiple orderings of pairs to find a valid path assignment."""
    bs = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    
    # Try natural order
    r = _try_solve(adj, branches, pairs)
    if r is not None:
        return r
    
    # Try sorted by shortest path
    order = _order_pairs_by_shortest(adj, branches)
    r = _try_solve(adj, branches, order)
    if r is not None:
        return r
    
    # Try reverse shortest
    r = _try_solve(adj, branches, list(reversed(order)))
    if r is not None:
        return r
    
    # Try with longer paths allowed
    r = _try_solve(adj, branches, pairs, use_long_paths=True)
    if r is not None:
        return r
    
    # Try backtracking with candidate paths
    if time.time() < deadline - 1:
        r = _try_solve_backtrack(adj, branches, deadline)
        if r is not None:
            return r
    
    # Try random orders (limited by time)
    for _ in range(15):
        if time.time() > deadline:
            break
        random.shuffle(pairs)
        r = _try_solve(adj, branches, pairs)
        if r is not None:
            return r
        # Also try with long paths
        r = _try_solve(adj, branches, pairs, use_long_paths=True)
        if r is not None:
            return r
    
    return None
>>>>>>> REPLACE

