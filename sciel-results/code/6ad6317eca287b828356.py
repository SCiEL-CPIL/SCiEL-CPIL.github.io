def construct():
    ...
    res13 = _try_rank13()
    if res13 is not None:
        ...
    U, V, W = _build_rank14()
    return {"U": U, "V": V, "W": W}
