    if time.time() > deadline:
        return _schoolbook()
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return _schoolbook()
