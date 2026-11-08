def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    U, V, W = _construct_rank_50()
    if time.time() > deadline:
        U, V, W = _construct_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook()
    return {"U": U, "V": V, "W": W}
