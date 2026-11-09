<<<<<<< SEARCH
def construct():
    t_start = time.time()
    rng = np.random.default_rng(12345)
    best = None
    best_val = -np.inf

    aspects = [1.0, 1.3, 0.77, 1.6, 0.62, 1.15, 0.87]
    starts_per_aspect = 3

    for a in aspects:
        for s in range(starts_per_aspect):
            if time.time() > _DEADLINE:
                break
            # mix: one near-grid start, rest random
            if s == 0:
                x0 = _seed_grid(a, rng)
            else:
                x0 = None
            # run solve (internally handles random x0 if x0 is None by reusing logic)
            if x0 is None:
                # build random x0 here
                w, h = _rect_dims(a)
                x0 = np.empty(3 * N)
                for k in range(N):
                    x0[2 * k] = rng.uniform(0.05, 0.95) * w
                    x0[2 * k + 1] = rng.uniform(0.05, 0.95) * h
                    x0[2 * N + k] = rng.uniform(0.02, 0.12)
            arr = _solve_aspect_x0(a, x0, rng)
            if arr is not None:
                val = np.sum(arr[:, 2])
                if val > best_val:
                    best_val = val
                    best = arr
        if time.time() > _DEADLINE:
            break

    if best is None:
        # fallback: seed-like uniform grid (always feasible)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best
=======
def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_val = -np.inf

    # Focus on promising aspect ratios
    aspects = [1.0, 1.2, 0.83, 1.4, 0.71, 1.1, 0.9, 1.3, 0.77]
    
    for a in aspects:
        if time.time() > _DEADLINE:
            break
        w, h = _rect_dims(a)
        
        # Try multiple diverse starting strategies
        start_configs = []
        
        # Grid-based starts with different col/row arrangements
        for cols in [3, 4, 5]:
            rows = int(np.ceil(N / cols))
            start_configs.append(_make_grid_start(a, cols, rows))
        
        # Random starts
        for _ in range(2):
            start_configs.append(_make_random_start(a, rng))
        
        # Clustered starts
        for nc in [2, 3, 4]:
            start_configs.append(_make_cluster_start(a, rng, nc))
        
        for x0 in start_configs:
            if time.time() > _DEADLINE:
                break
            arr = _solve_aspect_x0(a, x0, rng, max_iter=250)
            if arr is not None:
                val = np.sum(arr[:, 2])
                if val > best_val:
                    best_val = val
                    best = arr
    
    if best is None:
        # fallback: simple uniform grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9))
        return np.array(cells[:N])
    return best
>>>>>>> REPLACE

