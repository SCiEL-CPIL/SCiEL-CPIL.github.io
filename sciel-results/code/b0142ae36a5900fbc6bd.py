<<<<<<< SEARCH
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # First, try to find a rank-17 decomposition
    result = _try_rank17_search(n, m, p, T, deadline)
    if result is not None:
        U_l, V_l, W_l = result
        return {"U": U_l, "V": V_l, "W": W_l}
    
    # Strategy 2: random search for rank-17
    if time.time() < deadline - 30:
        result = _try_random_rank17(n, m, p, T, deadline)
        if result is not None:
            U_l, V_l, W_l = result
            return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fall back to known rank-18 solution
    U, V, W = _build_rank18(n, m, p)
    return {"U": U, "V": V, "W": W}
=======
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # First, try to find a rank-16 decomposition (Strassen 2x2 + 2x3)
    result = _try_rank16_search(n, m, p, T, deadline)
    if result is not None:
        U_l, V_l, W_l = result
        return {"U": U_l, "V": V_l, "W": W_l}
    
    # Try to find a rank-17 decomposition
    if time.time() < deadline - 20:
        result = _try_rank17_search(n, m, p, T, deadline)
        if result is not None:
            U_l, V_l, W_l = result
            return {"U": U_l, "V": V_l, "W": W_l}
    
    # Strategy 2: random search for rank-17
    if time.time() < deadline - 10:
        result = _try_random_rank17(n, m, p, T, deadline)
        if result is not None:
            U_l, V_l, W_l = result
            return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fall back to known rank-18 solution
    U, V, W = _build_rank18(n, m, p)
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
