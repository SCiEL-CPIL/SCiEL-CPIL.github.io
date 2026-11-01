<<<<<<< SEARCH
    # Last resort: return a very large LUW graph without verification (should not happen)
    n_luw, edges_luw = _luw_construct(3, g, d_override=15)
    return n_luw, edges_luw
=======
    # Last resort: return a large LUW graph (d=12) without verification (should not happen)
    n_luw, edges_luw = _luw_construct(3, g, d_override=12)
    return n_luw, edges_luw
>>>>>>> REPLACE
