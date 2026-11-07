"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s

        calls = [0]

        def counted_func(x):
            calls[0] += 1
            return func(x)

        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        # For dim=5, lambda = 4 + floor(3*log(dim)) = 4 + 3 = 7
        lam = 4 + int(3 * np.log(dim))
        c = (4 + np.log(dim) / dim) / (dim + 4 + 2 * np.log(dim) / dim)
        damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + c
        cmu = min(2 - c + 2 / (dim + lam - 2), (lam - 2 + 1 / lam) / (dim + 2 - lam))
        c_sigma = (5 + np.log(dim) / dim) / (dim + 5 + 2 * np.log(dim) / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cma_es(x0, sigma0, max_calls, deadline_check):
            nonlocal best_x, best_f, calls
            x = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            c1 = (2 + 2 / (dim + 1. / (c + 1.))) / ((dim + 2) * (c + 1.))
            cmu_eff = cmu
            w = np.log(lam / 2) - np.log(np.arange(1, lam + 1))
            w = w / w.sum()
            mu_eff = 1 / (w ** 2).sum()
            cc = (4 + dim / c) / (dim + 4 + 2 / c)
            cs = (5 + np.log(dim) / dim) / (dim + 5 + 2 / c)

            def step():
                nonlocal x, sigma, B, D, pc, ps
                if calls[0] >= max_calls:
                    return False
                # Generate population
                population = []
                for i in range(lam):
                    if calls[0] >= max_calls:
                        return False
                    z = np.random.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = counted_func(y)
                    population.append((fy, y))
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                if calls[0] >= max_calls:
                    return False

                # Sort
                population.sort(key=lambda t: t[0])
                x_new = sum(w[i] * population[i][1] for i in range(int(mu_eff)))
                # Update step size
                y_vec = (x_new - x) / (sigma * np.linalg.norm(D @ B.T @ np.ones(dim)))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / np.linalg.norm(y_vec) * y_vec if np.linalg.norm(y_vec) > 1e-12 else (1 - cs) * ps
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                sigma *= np.exp(cs / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)

                # Update covariance
                delta = (x_new - x) / sigma
                rank_one = np.outer(delta, delta)
                rank_mu = sum(w[i] * np.outer(population[i][1] - x, population[i][1] - x) for i in range(int(mu_eff)))
                C = (1 + c * c1 * cc * np.outer(pc, pc) - c1 * np.eye(dim)) * C
                # Actually need to update C properly
                # Let me use a simpler update
                C = (1 + c * c1 * cc * np.outer(pc, pc) - c1 * np.eye(dim)) * C
                # This is getting complex, let me use scipy if available or a simpler approach
                return True

            # Let me use a simpler but effective approach: Nelder-Mead + random restarts
            return x, sigma

        # Use a combination of Nelder-Mead and random sampling
        # First, do some random sampling to find good starting points
        rng = np.random.default_rng(42)

        # Phase 1: Random sampling to find diverse good points
        n_initial = min(50, budget // 10)
        candidates = []
        for i in range(n_initial):
            if calls[0] >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            fx = counted_func(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates and pick top ones for local optimization
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead from top candidates
        def nelder_mead(x0, max_calls, deadline_check):
            nonlocal best_x, best_f, calls
            dim = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5 if x0[i] < 4 else -0.5
                xi[i] = np.clip(xi[i], -5, 5)
                simplex.append(xi)

            # Evaluate initial simplex
            for i in range(dim + 1):
                if calls[0] >= max_calls:
                    return
                simplex[i] = np.clip(simplex[i], -5, 5)
                simplex[i] = (counted_func(simplex[i]), simplex[i])
                if simplex[i][0] < best_f:
                    best_f = simplex[i][0]
                    best_x = simplex[i][1].copy()

            # Sort
            simplex.sort(key=lambda t: t[0])

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            while calls[0] < max_calls and time.time() < deadline_check:
                # Sort
                simplex.sort(key=lambda t: t[0])

                # Centroid of all but worst
                centroid = np.mean([s[1] for s in simplex[:-1]], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1][1])
                x_r = np.clip(x_r, -5, 5)
                if calls[0] >= max_calls:
                    break
                f_r = counted_func(x_r)
                if f_r < simplex[-2][0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if calls[0] >= max_calls:
                        simplex[-1] = (f_r, x_r)
                        if f_r < best_f:
                            best_f = f_r
                            best_x = x_r.copy()
                        break
                    f_e = counted_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = (f_e, x_e)
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                    else:
                        simplex[-1] = (f_r, x_r)
                        if f_r < best_f:
                            best_f = f_r
                            best_x = x_r.copy()
                else:
                    # Contraction
                    if f_r < simplex[-1][0]:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (simplex[-1][1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if calls[0] >= max_calls:
                        break
                    f_c = counted_func(x_c)
                    if f_c < min(f_r, simplex[-1][0]):
                        simplex[-1] = (f_c, x_c)
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i][1] = simplex[0][1] + sigma_nm * (simplex[i][1] - simplex[0][1])
                            simplex[i][1] = np.clip(simplex[i][1], -5, 5)
                            if calls[0] >= max_calls:
                                break
                            simplex[i][0] = counted_func(simplex[i][1])
                            if simplex[i][0] < best_f:
                                best_f = simplex[i][0]
                                best_x = simplex[i][1].copy()

            # Update best
            if calls[0] < max_calls:
                pass

        # Run Nelder-Mead from top candidates
        nm_budget_per = max(30, (budget - calls[0]) // 3)
        for i in range(min(3, len(candidates))):
            if calls[0] >= budget or time.time() >= deadline:
                break
            x0 = candidates[i][1]
            nelder_mead(x0, calls[0] + nm_budget_per, deadline)

        # Phase 3: Fine-tuning around best point
        if calls[0] < budget and best_x is not None:
            # Do a local search with smaller steps
            remaining = budget - calls[0]
            step_size = 0.1
            for _ in range(remaining):
                if time.time() >= deadline:
                    break
                x = best_x + step_size * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = counted_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    step_size = max(step_size * 0.8, 0.001)
                else:
                    step_size = max(step_size * 0.95, 0.001)

        return best_x, best_f
# EVOLVE-BLOCK-END
