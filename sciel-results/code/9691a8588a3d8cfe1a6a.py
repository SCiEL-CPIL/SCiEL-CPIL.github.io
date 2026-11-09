    def run_local_search(init_centers, init_sides, max_time):
        nonlocal best_centers, best_sides, best_obj
        rng = np.random.default_rng()
        # ...
        dx = rng.normal(0, step_size, (n, 2))
        ds = rng.normal(0, step_size * 0.5, n)
        # ...
        if rng.random() < prob:
