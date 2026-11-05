<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin under 150s cap

    n, m, p = DIMENSIONS

    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        U, V, W, rank = _build_trivial(n, m, p)
    else:
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin under 150s cap

    n, m, p = DIMENSIONS

    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        U, V, W, rank = _build_trivial(n, m, p)
    else:
        # First try direct rank reduction
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline)
        
        # Then try slice-mixing with output basis changes
        if time.time() < deadline - 40:
            U, V, W, rank = _try_slice_mixing(n, m, p, U, V, W, deadline)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
