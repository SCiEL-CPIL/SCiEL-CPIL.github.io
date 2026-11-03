def construct():
    try:
        deadline = time.time() + 140
        result = _try_rank18_partition()
        if result is not None:
            return result
        return _make_schoolbook()
    except Exception:
        return _make_schoolbook()
