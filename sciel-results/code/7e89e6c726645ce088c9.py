<<<<<<< SEARCH
    # Random hyperplane projections of the embedding
    rng = np.random.default_rng(12345)
    n_proj = 64
    for i in range(n_proj):
        if time.time() > deadline:
            break
        r = rng.standard_normal(k)
        r /= (np.linalg.norm(r) + 1e-12)
        coords = top @ r
        s = np.where(coords >= 0, 1.0, -1.0)
        try_partition(s)

    # Also try a few pure random restarts as a safety net
    for i in range(16):
        if time.time() > deadline:
            break
        s = rng.choice([-1.0, 1.0], size=n)
        try_partition(s)
=======
    # Random hyperplane projections of the embedding
    rng = np.random.default_rng(12345)
    n_proj = 32  # Reduced from 64 to ensure time safety
    for i in range(n_proj):
        if time.time() > deadline:
            break
        r = rng.standard_normal(k)
        r /= (np.linalg.norm(r) + 1e-12)
        coords = top @ r
        s = np.where(coords >= 0, 1.0, -1.0)
        try_partition(s)

    # Also try a few pure random restarts as a safety net
    for i in range(16):
        if time.time() > deadline:
            break
        s = rng.choice([-1.0, 1.0], size=n)
        try_partition(s)
>>>>>>> REPLACE
