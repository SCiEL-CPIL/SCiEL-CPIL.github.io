<<<<<<< SEARCH
def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_params = None
    best_score = -np.inf

    # Define bounds for the 4 symmetric parameters
    # a1, a2 in [0, 0.5]
    # b1, b2 in [0, H]
    bounds = [(0.0, 0.5), (0.0, H), (0.0, 0.5), (0.0, H)]

    # Phase 1: Differential Evolution on the reduced 4D space
    try:
        result = differential_evolution(
            _objective_sym,
            bounds=bounds,
            maxiter=200,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=False,
            updating='immediate',
            init='sobol'
        )
        if time.time() > deadline:
            pass # Just proceed to return best found so far, but DE usually finishes fast
        best_params = result.x
        best_score = -result.fun
    except Exception:
        pass

    # Phase 2: Local polish with Nelder-Mead on the 4D space
    if time.time() < deadline - 10.0 and best_params is not None:
        try:
            res2 = minimize(
                _objective_sym,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if -res2.fun > best_score:
                best_params = res2.x
                best_score = -res2.fun
        except Exception:
            pass

    # Phase 3: Multiple random restarts of local polish if time remains
    if time.time() < deadline - 5.0:
        rng = np.random.RandomState(123)
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            params = rng.uniform(0, 0.5, size=2)
            params[1] = rng.uniform(0, H)
            params[3] = rng.uniform(0, H)
            # Ensure b < sqrt(3)*(0.5-a) roughly, though penalty handles it
            try:
                res = minimize(
                    _objective_sym,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if -res.fun > best_score:
                    best_params = res.x
                    best_score = -res2.fun # Correction: best_score = -res.fun
                    best_score = -res.fun
            except Exception:
                pass

    if best_params is not None:
        points = _symmetric_points(best_params)
        # Final safety: ensure all points are strictly inside
        for i in range(N):
            if not _is_inside(points[i]):
                # Nudge inward towards centroid
                points[i] = 0.95 * points[i] + 0.05 * np.array([0.5, H / 3])
        return points
    else:
        # Absolute fallback: return the fixed symmetric points
        points = _symmetric_points([0.1, 0.2, 0.2, 0.4])
        return points
=======
def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_params = None
    best_score = -np.inf

    # Define bounds for the 4 symmetric parameters
    # a1, a2 in [0, 0.5]
    # b1, b2 in [0, H]
    bounds = [(0.0, 0.5), (0.0, H), (0.0, 0.5), (0.0, H)]

    # Strategy: Multi-start Nelder-Mead
    # We will generate a set of initial points and optimize from each.
    # This is more robust than DE in this constrained, low-dim space.
    
    # Initial points:
    # 1. Center of bounds
    # 2. Random samples
    # 3. Specific heuristic guesses (e.g., grid points)
    
    rng = np.random.RandomState(42)
    initial_guesses = []
    
    # Center
    initial_guesses.append(np.array([0.25, H/2, 0.25, H/2]))
    
    # Grid-like guesses
    for a1 in [0.1, 0.2, 0.3]:
        for a2 in [0.1, 0.2, 0.3]:
            # Max b for given a is sqrt(3)*(0.5-a)
            max_b1 = np.sqrt(3) * (0.5 - a1)
            max_b2 = np.sqrt(3) * (0.5 - a2)
            b1 = max_b1 * 0.5
            b2 = max_b2 * 0.5
            initial_guesses.append(np.array([a1, b1, a2, b2]))
    
    # Random guesses
    for _ in range(20):
        a1 = rng.uniform(0.05, 0.4)
        a2 = rng.uniform(0.05, 0.4)
        max_b1 = np.sqrt(3) * (0.5 - a1)
        max_b2 = np.sqrt(3) * (0.5 - a2)
        b1 = rng.uniform(0.1, max_b1) if max_b1 > 0.1 else 0.0
        b2 = rng.uniform(0.1, max_b2) if max_b2 > 0.1 else 0.0
        initial_guesses.append(np.array([a1, b1, a2, b2]))

    for guess in initial_guesses:
        if time.time() > deadline - 10.0:
            break
        
        try:
            res = minimize(
                _objective_sym,
                guess,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue

    if best_params is not None:
        points = _symmetric_points(best_params)
        # Final safety: ensure all points are strictly inside
        for i in range(N):
            if not _is_inside(points[i]):
                # Nudge inward towards centroid
                points[i] = 0.95 * points[i] + 0.05 * np.array([0.5, H / 3])
        return points
    else:
        # Absolute fallback: return the fixed symmetric points
        points = _symmetric_points([0.1, 0.2, 0.2, 0.4])
        return points
>>>>>>> REPLACE
