<<<<<<< SEARCH
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U: shape (6, rank)
              V: shape (9, rank) 
              W: shape (6, rank)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Try ranks from 10 up to 18
    # Start with smaller ranks and increase
    for rank in range(10, N * M * P + 1):
        if time.time() > deadline:
            break
        
        # Try with integer entries first
        result = _search_rank(rank, {-1.0, 0.0, 1.0}, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try with half-integer entries
        result = _search_rank(rank, {-1.0, -0.5, 0.0, 0.5, 1.0}, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank-18 decomposition
    U, V, W = _try_rank18()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U: shape (6, rank)
              V: shape (9, rank) 
              W: shape (6, rank)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Try the rank-6 construction first
    result = _rank17_strassen()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank-18 decomposition
    U, V, W = _try_rank18()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE

