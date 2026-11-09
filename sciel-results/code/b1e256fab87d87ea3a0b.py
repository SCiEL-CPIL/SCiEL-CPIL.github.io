<<<<<<< SEARCH
def _nelder_mead(objective, x0, bounds, maxiter=200, timeout=None, start_time=None):
    """Nelder-Mead simplex optimization with bounds."""
    if start_time is None:
        start_time = time.time()

    n = len(x0)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 1 else 0.1
        xi[i] += step
        xi[i] = np.clip(xi[i], bounds[i][0], bounds[i][1])
        simplex.append(xi)

    fvals = np.array([objective(simplex[i]) for i in range(n + 1)])
    best_idx = np.argmin(fvals)
    fbest = fvals[best_idx]
    xbest = simplex[best_idx].copy()

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for _ in range(maxiter):
        if timeout and time.time() - start_time > timeout:
            break

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = fvals[order]

        if fvals[0] < fbest:
            fbest = fvals[0]
            xbest = simplex[0].copy()

        # Check convergence
        if np.allclose(fvals, fvals[0], rtol=1e-6, atol=1e-6):
            break

        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = objective(x_r)

        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (centroid - worst)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (worst - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = objective(simplex[i])

    return xbest, fbest
=======
def _nelder_mead(objective, x0, bounds, maxiter=200, timeout=None, start_time=None):
    """Nelder-Mead simplex optimization with bounds."""
    if start_time is None:
        start_time = time.time()

    n = len(x0)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 1 else 0.1
        xi[i] += step
        xi[i] = np.clip(xi[i], bounds[i][0], bounds[i][1])
        simplex.append(xi)

    fvals = np.array([objective(simplex[i]) for i in range(n + 1)])
    best_idx = np.argmin(fvals)
    fbest = fvals[best_idx]
    xbest = simplex[best_idx].copy()

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for _ in range(maxiter):
        if timeout and time.time() - start_time > timeout:
            break

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = fvals[order]

        if fvals[0] < fbest:
            fbest = fvals[0]
            xbest = simplex[0].copy()

        # Check convergence
        if np.allclose(fvals, fvals[0], rtol=1e-6, atol=1e-6):
            break

        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = objective(x_r)

        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (centroid - worst)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (worst - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = objective(simplex[i])

    return xbest, fbest


def _de_global_search(objective, bounds, popsize=15, maxiter=50, timeout=None, start_time=None):
    """
    Differential evolution global search.
    Returns (best_x, best_f).
    """
    if start_time is None:
        start_time = time.time()

    n = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    def wrapper(x):
        if timeout and time.time() - start_time > timeout:
            return 1e10
        return objective(x)

    try:
        result = differential_evolution(
            wrapper, bounds, popsize=popsize, maxiter=maxiter,
            tol=1e-6, seed=42, polish=False
        )
        if result.success or result.fun < 1e10:
            return result.x, result.fun
    except Exception:
        pass

    # Fallback: random search
    best_x = None
    best_f = float('inf')
    n_iter = 200
    for _ in range(n_iter):
        if timeout and time.time() - start_time > timeout:
            break
        x = lower + np.random.rand(n) * (upper - lower)
        f = wrapper(x)
        if f < best_f:
            best_f = f
            best_x = x

    if best_x is None:
        best_x = lower + 0.5 * (upper - lower)
        best_f = wrapper(best_x)

    return best_x, best_f
>>>>>>> REPLACE

