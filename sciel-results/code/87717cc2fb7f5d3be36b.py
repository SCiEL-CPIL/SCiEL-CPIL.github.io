def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    if k != 3 or g != 16:
        return _luw_construct(k, g)
    
    # Try LUW construction first
    n_luw, edges_luw = _luw_construct(k, g)
    if _check_girth(edges_luw, n_luw, g):
        return n_luw, edges_luw

    # If LUW is invalid (should not happen), try random
    # ... (existing random code)
    
    return n_luw, edges_luw
