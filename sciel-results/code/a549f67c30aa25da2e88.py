    # Try to extend if we have time
    if time.time() < deadline - 10.0:
        adj = _adj_bitsets(n, edges)
        k4s = _find_all_k4(adj, n)
        i4s = _find_all_i4(adj, n)
        
        # Try to extend up to 5 times (37 -> 42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            result = _try_extend(n, edges, k4s, i4s, deadline)
            if result is None:
                break
            new_n, new_edges, _, _ = result
            # Quick verification is done in _try_extend logic, but let's be safe
            # Actually, _try_extend guarantees no NEW K5/I5 if old graph was clean.
            # So we can skip full _is_55_free for speed, or do it once at the end.
            # But we need to update k4s and i4s for the next step.
            n, edges = new_n, new_edges
            adj = _adj_bitsets(n, edges)
            k4s = _find_all_k4(adj, n)
            i4s = _find_all_i4(adj, n)
            
    # Final verification
    res = _is_55_free(n, edges)
    if res is True:
        return (n, edges)
    else:
        # Fallback to known good 37
        return (37, _paley_edges(37))
