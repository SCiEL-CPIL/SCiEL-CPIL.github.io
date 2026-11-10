<<<<<<< SEARCH
    # Candidates 42+: Random normalized vectors with different seeds
    rng = np.random.default_rng(12345)
    n_random = 50
    for _ in range(n_random):
        if time.time() > deadline - 60:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
=======
    # Candidates 42+: Random normalized vectors with different seeds
    # Also add structured candidates: concentrated on few modes
    rng = np.random.default_rng(12345)
    n_random = 80
    for _ in range(n_random):
        if time.time() > deadline - 60:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Structured: sparse combinations (pick 2-4 random modes with random signs)
    n_structured = 40
    for _ in range(n_structured):
        if time.time() > deadline - 60:
            break
        c_sparse = np.zeros(n_m)
        n_modes = rng.integers(2, 5)
        chosen = rng.choice(n_m, size=n_modes, replace=False)
        for idx in chosen:
            c_sparse[idx] = rng.choice([-1.0, 1.0])
        c_sparse /= np.linalg.norm(c_sparse)
        candidates.append(c_sparse)
>>>>>>> REPLACE

