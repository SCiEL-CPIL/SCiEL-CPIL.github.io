<<<<<<< SEARCH
def _binary_search_threshold(deadline):
    """
    Binary search for the minimum threshold K such that a Hamilton cycle exists
    with all edges having arrival index <= K.
    """
    max_idx = len(_EDGE_DATA)
    
    lo = _N  # Minimum possible is _N (need at least N edges)
    hi = max_idx
    
    best_cycle = None
    best_threshold = max_idx + 1
    
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        
        cycle = _find_hamiltonian_cycle_threshold(mid, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_threshold = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_cycle, best_threshold
=======
def _binary_search_threshold(deadline):
    """
    Binary search for the minimum threshold K such that a Hamilton cycle exists
    with all edges having arrival index <= K.
    Uses algebraic fingerprinting to quickly eliminate infeasible thresholds
    before committing to expensive DFS.
    """
    max_idx = len(_EDGE_DATA)
    
    lo = _N  # Minimum possible is _N (need at least N edges)
    hi = max_idx
    
    best_cycle = None
    best_threshold = max_idx + 1
    
    # Phase A: Use algebraic fingerprinting to narrow the range
    # FP returns False only when we're certain no Hamilton cycle exists
    fp_lo = lo
    fp_hi = hi
    while fp_lo < fp_hi:
        if deadline is not None and time.time() > deadline - 15:
            break
        mid = (fp_lo + fp_hi) // 2
        if _algebraic_fingerprint(mid, deadline):
            fp_hi = mid
        else:
            fp_lo = mid + 1
    
    # Phase B: Exact DFS in the narrowed range with small buffer
    search_lo = max(lo, fp_lo - 3)
    search_hi = min(hi, fp_hi + 3)
    
    while search_lo <= search_hi and time.time() < deadline - 5:
        mid = (search_lo + search_hi) // 2
        
        cycle = _find_hamiltonian_cycle_threshold(mid, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_threshold = mid
            search_hi = mid - 1
        else:
            search_lo = mid + 1
    
    return best_cycle, best_threshold
>>>>>>> REPLACE

