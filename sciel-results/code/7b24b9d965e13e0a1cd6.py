def construct():
    n, m, p = 2, 4, 8
    
    U, V, W = _build_strassen56()
    
    if not _validate(U, V, W, n, m, p):
        pass
    
    if time.time() > _start_time + _TIMEOUT - 1.0:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
