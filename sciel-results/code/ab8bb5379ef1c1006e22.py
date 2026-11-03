def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    candidates = []
    
    # 1. Block Strassen
    U_b1, V_b1, W_b1 = _strassen_construct(n, m, p)
    if _verify_fast(U_b1, V_b1, W_b1, n, m, p, atol=1e-9):
        candidates.append((len(U_b1[0]), {"U": U_b1, "V": V_b1, "W": W_b1}, "block"))
        
    # 2. Kronecker Strassen
    if time.time() < deadline - 30:
        U_b2, V_b2, W_b2 = _kron_strassen_construct(n, m, p)
        if _verify_fast(U_b2, V_b2, W_b2, n, m, p, atol=1e-9):
            candidates.append((len(U_b2[0]), {"U": U_b2, "V": V_b2, "W": W_b2}, "kron"))
            
    if not candidates:
        return _schoolbook()
        
    # Sort by initial rank
    candidates.sort(key=lambda x: x[0])
    
    best_result = candidates[0][1]
    best_rank = candidates[0][0]
    
    # Try to reduce each candidate
    for rank, res, name in candidates:
        if time.time() > deadline - 20:
            break
            
        # Only try to reduce if it's not the same as the current best (or if we want to try both)
        # We'll try to reduce the one with the highest potential (lowest initial rank first)
        
        reduced = _try_reduce_from_baseline(res["U"], res["V"], res["W"], n, m, p, deadline - 10)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            new_rank = len(U_r[0])
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                
    return best_result
