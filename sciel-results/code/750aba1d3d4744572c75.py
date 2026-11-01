# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Handle known small cages explicitly
    if (k, g) == (3, 5):
        return _petersen()
    elif (k, g) == (3, 6):
        return _heawood()
    elif (k, g) == (3, 7):
        return _tutte_coxeter()
    elif (k, g) == (3, 8):
        return _tutte_coxeter()
    elif (k, g) == (4, 5):
        return _klein_30()
    elif (k, g) == (4, 7):
        return _klein_56()
    elif (k, g) == (5, 9):
        # Use LUW construction which gives 1250 vertices.
        # This is a valid upper bound.
        return _luw_construct(5, 9)

    # 2. LUW Construction as baseline
    n_luw, edges_luw = _luw_construct(k, g)

    # 3. For small k and g, try random search to find smaller valid graphs
    # Only attempt if LUW is not too large and we have time
    if k <= 4 and g <= 10 and time.time() < deadline - 30:
        n_rand, edges_rand = _random_search(k, g, n_luw, deadline)
        if n_rand is not None and n_rand < n_luw:
            return n_rand, edges_rand

    return n_luw, edges_luw
# ... (rest of the functions remain the same)
