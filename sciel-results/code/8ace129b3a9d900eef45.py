    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).flatten()
            return float(np.clip(func(x), -1e15, 1e15))

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Generate diverse starting points
        n_starts = 8
        starts = [np.zeros(dim)]
        
        # Latin Hypercube Sampling
        perms = [rng.permutation(n_starts) for _ in range(dim)]
        for i in range(1, n_starts):
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = -5.0 + (perms[d][i] + 0.5) * (10.0 / n_starts)
            starts.append(x)

        # Phase 2: Local optimization from each start
        # Conservative maxiter to ensure we stay within budget
        # L-BFGS-B typically uses 1-2 function evals per iteration
        max_iter_local = 15
        
        for start in starts:
            if calls[0] >= budget - 30 or time.time() > deadline:
                break
            
            try:
                res = minimize(
                    wrapped,
                    start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max_iter_local, 'maxls': 20, 'gtol': 1e-5, 'ftol': 1e-8}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                # If budget exceeded or other error, stop
                break

        # Phase 3: Final Polish on the best point found
        if best_x is not None and calls[0] < budget - 10:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxls': 50, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapped(best_x)
            except:
                best_f = 0.0

        return best_x, best_f
