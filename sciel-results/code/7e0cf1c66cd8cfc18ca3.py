"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Parallel asynchronous Nelder-Mead simplex ensemble for Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize 5 simplexes: 4 corners + center
        # For dim=5, we use a simplified initialization
        # Corner points: combinations of -4 and 4
        # We'll pick 4 distinct corners and the center

        # Generate initial points
        # Center
        center = np.zeros(dim)

        # 4 corners (using random sign patterns for diversity)
        corners = []
        # Use fixed deterministic corners for reproducibility
        # Corner 1: all positive
        corners.append(np.full(dim, 4.0))
        # Corner 2: all negative
        corners.append(np.full(dim, -4.0))
        # Corner 3: alternating signs
        alt1 = np.array([4.0 if i % 2 == 0 else -4.0 for i in range(dim)])
        corners.append(alt1)
        # Corner 4: opposite alternating
        alt2 = np.array([-4.0 if i % 2 == 0 else 4.0 for i in range(dim)])
        corners.append(alt2)

        initial_points = [center] + corners
        n_simplexes = len(initial_points)

        # Each simplex has dim+1 vertices
        # Initialize each simplex around its starting point
        simplexes = []
        sigma_init = 1.0

        for start in initial_points:
            simplex = []
            simplex.append(start.copy())
            for i in range(dim):
                vertex = start.copy()
                vertex[i] += sigma_init
                simplex.append(vertex)
            simplex = np.array(simplex)
            simplex = np.clip(simplex, -5, 5)
            simplexes.append(simplex)

        # Evaluate all vertices
        best_x = None
        best_f = np.inf
        call_count = 0

        # Store evaluated points to avoid re-evaluation
        # Use a dictionary with rounded keys for approximate matching
        eval_cache = {}

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            # Round for cache key
            key = tuple(np.round(x, 6))
            if key in eval_cache:
                return eval_cache[key]
            if call_count >= budget:
                return None
            fx = func(x)
            call_count += 1
            eval_cache[key] = fx
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Evaluate all initial vertices
        for s in simplexes:
            for v in s:
                fx = evaluate(v)
                if fx is None:
                    return best_x, best_f

        # Track best per simplex
        simplex_best_f = []
        simplex_best_idx = []
        for s in simplexes:
            fvals = [eval_cache[tuple(np.round(v, 6))] for v in s]
            best_idx = int(np.argmin(fvals))
            simplex_best_f.append(fvals[best_idx])
            simplex_best_idx.append(best_idx)

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        max_iter_per_simplex = 50

        # Main loop: alternate between simplexes based on progress
        # Use a priority queue approach: simplex with most recent improvement gets priority
        # But for simplicity, we'll round-robin with adaptive allocation

        # Track iterations per simplex
        simplex_iters = [0] * n_simplexes
        simplex_stagnation = [0] * n_simplexes
        simplex_best_history = simplex_best_f.copy()

        while call_count < budget:
            # Select next simplex: prefer those with recent improvement or low stagnation
            # Simple strategy: cycle through simplexes, but skip stagnant ones
            best_progress = -np.inf
            selected = 0
            for i in range(n_simplexes):
                progress = simplex_best_history[i] - simplex_best_f[i]
                if progress > best_progress:
                    best_progress = progress
                    selected = i
                # If all stagnant, just cycle
            # If no improvement detected, cycle round-robin
            if best_progress <= 1e-10:
                selected = simplex_iters.index(min(simplex_iters))

            simplex = simplexes[selected]
            dim_local = self.dim

            # Get current function values
            fvals = []
            for v in simplex:
                key = tuple(np.round(v, 6))
                if key in eval_cache:
                    fvals.append(eval_cache[key])
                else:
                    fx = evaluate(v)
                    if fx is None:
                        break
                    fvals.append(fx)
            else:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = [fvals[i] for i in order]

                # Nelder-Mead steps
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)
                if f_r is None:
                    break

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if f_e is None:
                        break
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
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c is None:
                        break
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim_local + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            key = tuple(np.round(simplex[i], 6))
                            if key not in eval_cache:
                                fx = evaluate(simplex[i])
                                if fx is None:
                                    break
                                fvals[i] = fx
                        if fvals[-1] is None:
                            break

                # Update simplex best
                new_best_idx = int(np.argmin(fvals))
                new_best_f = fvals[new_best_idx]
                if new_best_f < simplex_best_f[selected]:
                    simplex_best_f[selected] = new_best_f
                    simplex_stagnation[selected] = 0
                else:
                    simplex_stagnation[selected] += 1

                simplex_iters[selected] += 1

                # If simplex has converged, reset it with new random points
                if simplex_stagnation[selected] > 20:
                    # Reset simplex with new random location
                    new_center = rng.uniform(-4, 4, dim_local)
                    new_simplex = [new_center.copy()]
                    for i in range(dim_local):
                        vertex = new_center.copy()
                        vertex[i] += 0.5
                        new_simplex.append(vertex)
                    new_simplex = np.array(new_simplex)
                    new_simplex = np.clip(new_simplex, -5, 5)
                    simplexes[selected] = new_simplex
                    simplex_stagnation[selected] = 0
                    # Evaluate new vertices
                    for v in new_simplex:
                        fx = evaluate(v)
                        if fx is None:
                            break
                    if call_count >= budget:
                        break

                if call_count >= budget:
                    break

            # If budget exhausted, break
            if call_count >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
