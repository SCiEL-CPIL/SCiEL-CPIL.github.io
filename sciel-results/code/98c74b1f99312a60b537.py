<<<<<<< SEARCH
    # Try rank 32 first
    U, V, W = build_rank32()
    U = sanitize_matrix(U)
    V = sanitize_matrix(V)
    W = sanitize_matrix(W)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank 32 fails (shouldn't happen), try rank 36 (schoolbook)
    if time.time() > deadline:
        # If we're out of time, we have to return something. 
        # Returning an invalid result scores 0. 
        # Returning a valid result scores > 0.
        # We haven't verified rank 36 yet. 
        # But rank 32 construction is deterministic and fast. 
        # If verify failed, it's a bug in my code.
        pass

    U36, V36, W36 = build_rank63() # This is actually rank 36
    U36 = sanitize_matrix(U36)
    V36 = sanitize_matrix(V36)
    W36 = sanitize_matrix(W36)
    
    if verify(U36, V36, W36):
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    # If both fail, raise error (scores 0)
    raise ValueError("Failed to construct valid decomposition")
=======
    # Try rank 32 first
    U, V, W = build_rank32()
    U = sanitize_matrix(U)
    V = sanitize_matrix(V)
    W = sanitize_matrix(W)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank 32 fails, try rank 36 (schoolbook) as fallback
    U36, V36, W36 = build_rank63() # This is actually rank 36
    U36 = sanitize_matrix(U36)
    V36 = sanitize_matrix(V36)
    W36 = sanitize_matrix(W36)
    
    if verify(U36, V36, W36):
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    # If both fail, raise error (scores 0)
    raise ValueError("Failed to construct valid decomposition")
>>>>>>> REPLACE
