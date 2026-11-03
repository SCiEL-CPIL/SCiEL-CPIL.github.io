<<<<<<< SEARCH
def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}

    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("circ", lambda: _circulant_search(k, g)),
        ("rand0", lambda: _random_search(k, g, seed=0)),
        ("rand1", lambda: _random_search(k, g, seed=1)),
        ("rand2", lambda: _random_search(k, g, seed=2)),
        ("rand3", lambda: _random_search(k, g, seed=3)),
    ]

    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, k, g), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 3.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.05)

    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
=======
def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}

    # First try known constructions (fast, deterministic)
    def _try_known():
        res = _known_constructions(k, g)
        if res is not None:
            n, edges = res
            if _is_valid(_build_adj(edges, n), n, k, g):
                return (n, edges)
        return None

    jobs = [
        ("known", _try_known),
        ("circ", lambda: _circulant_search(k, g)),
        ("rand0", lambda: _random_search(k, g, seed=0)),
        ("rand1", lambda: _random_search(k, g, seed=1)),
        ("rand2", lambda: _random_search(k, g, seed=2)),
        ("rand3", lambda: _random_search(k, g, seed=3)),
    ]

    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, k, g), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 5.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.05)

    best = holder.get("best")
    if best is None:
        # Fallback: complete graph (always valid for g<=3)
        n = k + 1
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        best = (n, edges)
    return best
>>>>>>> REPLACE

