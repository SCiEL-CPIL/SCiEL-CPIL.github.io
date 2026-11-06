<<<<<<< SEARCH
def construct():
    """
    Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j) where 0 <= i < j < 18.
    """
    global _start_time
    _start_time = time.time()
    
    # Generate initial candidate networks
    candidates = []
    
    # Candidate 1: Odd-even transposition (153 comparators)
    candidates.append(_odd_even_transposition(N))
    
    # Candidate 2: Bitonic 16 + merge 2
    if _check_time() < _DEADLINE - 10:
        try:
            c2 = _construct_bitonic_18()
            if _is_valid_network(c2):
                candidates.append(c2)
        except Exception:
            pass
    
    # Candidate 3: Hybrid
    if _check_time() < _DEADLINE - 10:
        try:
            c3 = _construct_hybrid()
            if _is_valid_network(c3):
                candidates.append(c3)
        except Exception:
            pass
    
    # Pick the shortest valid candidate
    if not candidates:
        # Fallback: return odd-even transposition
        return _odd_even_transposition(N)
    
    candidates.sort(key=len)
    best = candidates[0]
    
    # Local search to prune further
    if _check_time() < _DEADLINE - 5:
        best = _local_search(best, time_limit=_DEADLINE - _check_time() - 5)
    
    # Final validity check
    if not _is_valid_network(best):
        return _odd_even_transposition(N)
    
    return best
# EVOLVE-BLOCK-END
=======
def construct():
    """
    Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j) where 0 <= i < j < 18.
    """
    global _start_time
    _start_time = time.time()
    
    # Generate initial candidate networks
    candidates = []
    
    # Candidate 1: Odd-even transposition (153 comparators)
    candidates.append(_odd_even_transposition(N))
    
    # Candidate 2: Bitonic 16 + merge 2
    if _check_time() < _DEADLINE - 10:
        try:
            c2 = _construct_bitonic_18()
            if _is_valid_network(c2):
                candidates.append(c2)
        except Exception:
            pass
    
    # Candidate 3: Hybrid
    if _check_time() < _DEADLINE - 10:
        try:
            c3 = _construct_hybrid()
            if _is_valid_network(c3):
                candidates.append(c3)
        except Exception:
            pass
    
    # Pick the shortest valid candidate
    if not candidates:
        # Fallback: return odd-even transposition
        return _odd_even_transposition(N)
    
    candidates.sort(key=len)
    best = candidates[0]
    
    # Local search to prune further
    if _check_time() < _DEADLINE - 5:
        best = _local_search(best, time_limit=_DEADLINE - _check_time() - 5)
    
    # Final validity check
    if not _is_valid_network(best):
        return _odd_even_transposition(N)
    
    return best
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

