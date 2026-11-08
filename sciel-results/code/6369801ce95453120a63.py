def construct():
    ...
    h = np.full(N, 0.5, dtype=np.float64)
    if _elapsed() > _DEADLINE:
        return h
    if not _valid(h):
        h = np.full(N, 0.5, dtype=np.float64)
    return h
