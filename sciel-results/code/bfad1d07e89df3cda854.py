<<<<<<< SEARCH
def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # 1D search over alpha for generalized Laguerre nodes
    # roots_genlaguerre(n, alpha) gives nodes of L_n^(alpha)(x)
    # alpha = 0 is standard Laguerre
    best_roots = None
    best_c4 = np.inf

    # Scan alpha values around 0
    alpha_values = [0.0]
    # Add values on a log-spaced grid around 0, and some linear values
    for a in np.linspace(-5.0, 5.0, 51):
        alpha_values.append(a)
    # Also try some specific values
    alpha_values.extend([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 0.01, -0.01])

    alpha_values = sorted(set(alpha_values))

    for alpha in alpha_values:
        if not time_ok():
            break
        try:
            nodes, _ = roots_genlaguerre(12, alpha)
            nodes = np.sort(nodes)
            if np.any(nodes <= 0):
                continue
            c4 = _compute_c4(nodes)
            if np.isfinite(c4) and c4 < best_c4:
                best_c4 = c4
                best_roots = nodes.copy()
        except Exception:
            continue

    # If no valid roots found, fallback to standard Laguerre
    if best_roots is None:
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)

    # Local refinement from the best alpha point
    try:
        res = minimize(
            _objective_grad,
            best_roots.copy(),
            jac=True,
            method='L-BFGS-B',
            bounds=[(1e-8, None)] * 12,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if np.all(res.x > 0) and np.isfinite(res.fun):
            candidate = np.sort(res.x)
            c4 = _compute_c4(candidate)
            if c4 < best_c4:
                best_c4 = c4
                best_roots = candidate.copy()
    except Exception:
        pass

    # Additional refinement from nearby alpha values
    if time_ok():
        # Find which alpha gave the best result and try neighbors
        # Re-scan a finer grid around the best alpha
        # First, find the best alpha from our scan
        best_alpha = 0.0
        best_alpha_c4 = np.inf
        for alpha in alpha_values:
            try:
                nodes, _ = roots_genlaguerre(12, alpha)
                nodes = np.sort(nodes)
                if np.any(nodes <= 0):
                    continue
                c4 = _compute_c4(nodes)
                if np.isfinite(c4) and c4 < best_alpha_c4:
                    best_alpha_c4 = c4
                    best_alpha = alpha
            except Exception:
                continue

        # Fine scan around best alpha
        fine_alphas = np.linspace(best_alpha - 0.5, best_alpha + 0.5, 21)
        for alpha in fine_alphas:
            if not time_ok():
                break
            try:
                nodes, _ = roots_genlaguerre(12, alpha)
                nodes = np.sort(nodes)
                if np.any(nodes <= 0):
                    continue
                c4 = _compute_c4(nodes)
                if np.isfinite(c4) and c4 < best_c4:
                    best_c4 = c4
                    best_roots = nodes.copy()
                    # Refine this point
                    try:
                        res = minimize(
                            _objective_grad,
                            best_roots.copy(),
                            jac=True,
                            method='L-BFGS-B',
                            bounds=[(1e-8, None)] * 12,
                            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                        )
                        if np.all(res.x > 0) and np.isfinite(res.fun):
                            candidate = np.sort(res.x)
                            c42 = _compute_c4(candidate)
                            if c42 < best_c4:
                                best_c4 = c42
                                best_roots = candidate.copy()
                    except Exception:
                        pass
            except Exception:
                continue

    # Final high-precision polish
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots.copy(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    return [float(xi) for xi in best_roots]
=======
def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 60.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    def elapsed():
        return time.time() - start

    # Initial good starting points from structured families
    initial_roots_list = []

    # Standard Laguerre nodes
    try:
        x0, _ = roots_laguerre(12)
        initial_roots_list.append(np.sort(x0))
    except Exception:
        pass

    # Generalized Laguerre for a few alpha values
    for alpha in [0.0, 0.5, -0.5, 1.0, -1.0, 2.0]:
        try:
            nodes, _ = roots_genlaguerre(12, alpha)
            nodes = np.sort(nodes)
            if np.all(nodes > 0):
                initial_roots_list.append(nodes)
        except Exception:
            pass

    # Geometric progression
    for ratio in [0.8, 0.9, 1.0, 1.1, 1.2]:
        base = 1.0
        roots = base * ratio ** np.arange(12)
        if np.all(roots > 0):
            initial_roots_list.append(roots)

    if not initial_roots_list:
        initial_roots_list = [np.sort(roots_laguerre(12)[0])]

    # Evaluate all initial points
    best_c4 = np.inf
    best_roots = None
    for r in initial_roots_list:
        try:
            c4 = _compute_c4(r)
            if np.isfinite(c4) and c4 < best_c4:
                best_c4 = c4
                best_roots = r.copy()
        except Exception:
            pass

    # Phase 1: Differential Evolution in log-space (global search)
    # Log-space bounds: roots roughly in [0.01, 100] -> log in [-4.6, 4.6]
    log_bounds = [(-4.6, 4.6)] * 12

    # Use the best initial point to seed DE
    best_log = np.log(best_roots) if best_roots is not None else np.zeros(12)

    # Run DE with a time budget
    de_budget = min(180.0, DEADLINE - SAFETY - elapsed())
    if de_budget > 30:
        try:
            def _de_obj(log_r):
                if elapsed() > DEADLINE - SAFETY:
                    return 1e10
                return _log_objective(log_r)

            de_res = differential_evolution(
                _de_obj,
                bounds=log_bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=-1
            )
            if de_res.fun < best_c4 and np.isfinite(de_res.fun):
                best_c4 = de_res.fun
                best_roots = np.sort(np.exp(de_res.x))
        except Exception:
            pass

    # Phase 2: L-BFGS-B local refinement from best point (in log-space)
    if time_ok() and best_roots is not None:
        try:
            log_best = np.log(best_roots)
            res = minimize(
                _log_objective_grad,
                log_best.copy(),
                jac=True,
                method='L-BFGS-B',
                bounds=log_bounds,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.isfinite(res.fun) and res.fun < best_c4:
                best_c4 = res.fun
                best_roots = np.sort(np.exp(res.x))
        except Exception:
            pass

    # Phase 3: Multi-start local refinement from all initial points + DE result
    if time_ok():
        candidates = list(initial_roots_list)
        if best_roots is not None:
            candidates.append(best_roots)

        for r in candidates:
            if not time_ok():
                break
            try:
                log_r = np.log(r)
                res = minimize(
                    _log_objective_grad,
                    log_r.copy(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=log_bounds,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 30}
                )
                if np.isfinite(res.fun) and res.fun < best_c4:
                    best_c4 = res.fun
                    best_roots = np.sort(np.exp(res.x))
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    if time_ok() and best_roots is not None:
        try:
            log_best = np.log(best_roots)
            res = minimize(
                _log_objective_grad,
                log_best.copy(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxls': 30}
            )
            if np.isfinite(res.fun) and res.fun < best_c4:
                best_c4 = res.fun
                best_roots = np.sort(np.exp(res.x))
        except Exception:
            pass

    # Safety: ensure all roots are positive and finite
    if best_roots is None or not np.all(np.isfinite(best_roots)) or np.any(best_roots <= 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
>>>>>>> REPLACE
