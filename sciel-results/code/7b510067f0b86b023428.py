def _bclr_444_valid(terms):
    """Runtime verification: does this 4x4x4 tensor reproduce A*B exactly?"""
    rng = np.random.default_rng(12345)
    for _ in range(3):
        A = rng.integers(-3, 4, size=(4, 4))
        B = rng.integers(-3, 4, size=(4, 4))
        C = A @ B
        acc = np.zeros((4, 4), dtype=object)
        for u, v, w in terms:
            um = np.array(u, dtype=object).reshape(4, 4)
            vm = np.array(v, dtype=object).reshape(4, 4)
            wm = np.array(w, dtype=object).reshape(4, 4)
            acc += um * (vm * wm)
        if not np.array_equal(acc, C):
            return False
    return True
