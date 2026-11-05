<<<<<<< SEARCH
    # For k=5, try PSL(2,11) coset graph first
    if k == 5 and g == 9:
        psl_result = _coset_graph_psl2_11()
        if psl_result is not None:
            n_psl, edges_psl = psl_result
            if _validate_graph(n_psl, edges_psl, k, g):
                return n_psl, edges_psl
    
    # General case: use LUW construction
    n, edges = _luw_construct(k, g)
    
    # Validate
    if _validate_graph(n, edges, k, g):
        # Try to improve via local search
        if not _check_deadline():
            n, edges = _edge_swap_improve(n, edges, k, g, rng)
        return n, edges
    
    # If LUW failed validation (shouldn't happen for valid q), try randomized
    # fallback: just return LUW as-is (it should be valid)
    return n, edges
=======
    # Try Cayley graph constructions first
    cayley_result = _cayley_graph_nonabelian(k, g)
    if cayley_result is not None:
        n_cayley, edges_cayley = cayley_result
        if _validate_graph(n_cayley, edges_cayley, k, g):
            return n_cayley, edges_cayley
    
    # General case: use LUW construction
    n, edges = _luw_construct(k, g)
    
    # Validate
    if _validate_graph(n, edges, k, g):
        # Try to improve via local search
        if not _check_deadline():
            n, edges = _edge_swap_improve(n, edges, k, g, rng)
        return n, edges
    
    # If LUW failed validation (shouldn't happen for valid q), try randomized
    # fallback: just return LUW as-is (it should be valid)
    return n, edges
>>>>>>> REPLACE
