def _make_hashable(x):
    if isinstance(x, list):
        return tuple(_make_hashable(i) for i in x)
    if isinstance(x, dict):
        return str(sorted(x.items()))
    return x
