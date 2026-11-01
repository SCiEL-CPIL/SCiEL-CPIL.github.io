<<<<<<< SEARCH
    # Try to find rank-23
    result = try_find_rank23()
    if result is not None:
        U, V, W = result
        if verify(U, V, W, 23):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank-27
    U, V, W = make_rank27()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    # Try to find rank-23
    result = try_find_rank23()
    if result is not None:
        U, V, W = result
        if verify(U, V, W, 23):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank-27
    U, V, W = make_rank27()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
