<<<<<<< SEARCH
    # Apply peeling iteratively
    U, V, W = fast_peel(U, V, W)
    
    # Second pass
    if time.time() < deadline - 25:
        U2, V2, W2 = fast_peel(U, V, W)
        if U2.shape[1] < U.shape[1]:
            U, V, W = U2, V2, W2
    
    # Try swap-based peeling
    if time.time() < deadline - 30:
        U3, V3, W3 = peel_with_swap(U, V, W)
        if U3.shape[1] < U.shape[1]:
            U, V, W = U3, V3, W3
=======
    # Apply peeling iteratively
    U, V, W = fast_peel(U, V, W)
    
    # Second pass
    if time.time() < deadline - 25:
        U2, V2, W2 = fast_peel(U, V, W)
        if U2.shape[1] < U.shape[1]:
            U, V, W = U2, V2, W2
    
    # Try local optimization peeling
    if time.time() < deadline - 30:
        U3, V3, W3 = peel_with_local_opt(U, V, W)
        if U3.shape[1] < U.shape[1]:
            U, V, W = U3, V3, W3
    
    # Another round of fast peeling after local opt
    if time.time() < deadline - 15:
        U4, V4, W4 = fast_peel(U, V, W)
        if U4.shape[1] < U.shape[1]:
            U, V, W = U4, V4, W4
>>>>>>> REPLACE

