"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


class Optimizer:
    """Parallel Best-First Nelder-Mead with Latin Hypercube initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Latin Hypercube sampling for initial points
        def latin_hypercube_sample(n_samples, dim, seed=42):
            rng = np.random.default_rng(seed)
            samples = []
            for i in range(n_samples):
                # Each dimension gets a unique interval
                intervals = np.linspace(-5, 5, dim + 1)
                x = np.zeros(dim)
                for d in range(dim):
                    # Assign a random interval to each dimension
                    idx = rng.integers(0, dim)
                    # But we need a permutation for proper LHS
                    # Let's do proper LHS
                # Proper LHS: for each dimension, assign a unique interval
                perm = rng.permutation(dim)
                for d in range(dim):
                    interval = perm[d]
                    x[d] = rng.uniform(intervals[interval], intervals[interval + 1])
                samples.append(x)
            return np.array(samples)

        # Initialize multiple simplexes using LHS
        n_simplexes = 5
        alpha = 0.3

        # Generate LHS points
        lhs_points = latin_hypercube_sample(n_simplexes, dim, seed=42)

        # Initialize each simplex
        simplex_list = []  # List of dicts: {'vertices': [...], 'fvals': [...], 'centroid': ..., 'worst_idx': ...}

        for s in range(n_simplexes):
            if not time_ok() or calls >= budget:
                break
            x0 = lhs_points[s]
            f0 = func(x0)
            calls += 1
            record(x0, f0)

            vertices = [x0.copy()]
            fvals = [f0]

            for i in range(dim):
                if not time_ok() or calls >= budget:
                    break
                xi = x0.copy()
                xi[i] += alpha
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                record(xi, fi)
                vertices.append(xi)
                fvals.append(fi)

            if len(vertices) < 2:
                continue

            simplex_list.append({
                'vertices': vertices,
                'fvals': fvals,
                's': s
            })

        if not simplex_list:
            return best_x, best_f

        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5

        # Priority queue: (priority, simplex_index)
        # Priority: lower is better (more promising)
        # We use the best function value in the simplex as priority
        pq = []

        def simplex_priority(sim):
            # Use the best function value in the simplex
            return min(sim['fvals'])

        for i, sim in enumerate(simplex_list):
            heapq.heappush(pq, (simplex_priority(sim), i))

        # Main loop: process the most promising simplex
        while calls < budget and time_ok():
            if not pq:
                break

            # Get the most promising simplex
            priority, sim_idx = heapq.heappop(pq)
            sim = simplex_list[sim_idx]

            # Sort simplex by function value
            indices = np.argsort(sim['fvals'])
            sim['vertices'] = [sim['vertices'][i] for i in indices]
            sim['fvals'] = [sim['fvals'][i] for i in indices]

            if len(sim['vertices']) < 2:
                continue

            x_best = sim['vertices'][0]
            f_best = sim['fvals'][0]
            x_worst = sim['vertices'][-1]
            f_worst = sim['fvals'][-1]
            x_second_worst = sim['vertices'][-2]
            f_second_worst = sim['fvals'][-2]

            # Centroid of all but worst
            centroid = np.mean(sim['vertices'][:-1], axis=0)

            # Reflection
            x_r = centroid + gamma * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            if not time_ok() or calls >= budget:
                break
            f_r = func(x_r)
            calls += 1
            record(x_r, f_r)

            if f_r < f_best:
                # Expansion
                x_e = centroid + rho * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                if not time_ok() or calls >= budget:
                    break
                f_e = func(x_e)
                calls += 1
                record(x_e, f_e)
                if f_e < f_r:
                    sim['vertices'][-1] = x_e
                    sim['fvals'][-1] = f_e
                else:
                    sim['vertices'][-1] = x_r
                    sim['fvals'][-1] = f_r
            elif f_r < f_second_worst:
                sim['vertices'][-1] = x_r
                sim['fvals'][-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_c = centroid + sigma_nm * (x_r - centroid)
                else:
                    x_c = centroid - sigma_nm * (x_worst - centroid)
                x_c = np.clip(x_c, -5, 5)

                if not time_ok() or calls >= budget:
                    break
                f_c = func(x_c)
                calls += 1
                record(x_c, f_c)

                if f_c < min(f_r, f_worst):
                    sim['vertices'][-1] = x_c
                    sim['fvals'][-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(sim['vertices'])):
                        if not time_ok() or calls >= budget:
                            break
                        sim['vertices'][i] = x_best + shrink * (sim['vertices'][i] - x_best)
                        sim['vertices'][i] = np.clip(sim['vertices'][i], -5, 5)
                        sim['fvals'][i] = func(sim['vertices'][i])
                        calls += 1
                        record(sim['vertices'][i], sim['fvals'][i])
                    if not time_ok():
                        break

            # Check for convergence of this simplex
            f_range = np.max(sim['fvals']) - np.min(sim['fvals'])

            # Re-prioritize this simplex
            if f_range < 1e-12:
                # Converged, don't re-add
                continue
            else:
                heapq.heappush(pq, (simplex_priority(sim), sim_idx))

        # Final polish: coordinate descent on best point
        if best_x is not None and time_ok() and calls < budget:
            x_polish = best_x.copy()
            f_polish = best_f
            step = 0.5
            for _ in range(20):
                if not time_ok() or calls >= budget:
                    break
                improved = False
                for d in range(dim):
                    for direction in [1, -1]:
                        x_trial = x_polish.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        if not time_ok() or calls >= budget:
                            break
                        f_trial = func(x_trial)
                        calls += 1
                        if f_trial < f_polish:
                            x_polish = x_trial
                            f_polish = f_trial
                            record(x_polish, f_polish)
                            improved = True
                    if not time_ok() or calls >= budget:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
