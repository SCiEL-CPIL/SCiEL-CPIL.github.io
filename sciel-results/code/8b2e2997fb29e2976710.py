"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel Tempering CMA-ES with Nelder-Mead polish for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = np.inf
        calls = 0

        # Phase 1: Quick exploration to find diverse starting points
        # Use a mix of deterministic and random starts
        starts = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            np.ones(dim) * 3.0,
            -np.ones(dim) * 3.0,
        ]
        for _ in range(15):
            starts.append(rng.uniform(-4, 4, dim))

        # Evaluate a subset of starts to find good initial points
        n_explore = 12
        for s in starts[:n_explore]:
            if calls >= 20 or time_left() < 5:
                break
            x0 = np.clip(s, -5, 5)
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: Parallel Tempering with 5 CMA-ES instances
        # Each instance has a different temperature scale (sigma multiplier)
        n_replicas = 5
        # Temperature scales: 1.0 (coldest/most precise), 1.5, 2.0, 3.0, 5.0 (hottest/most exploratory)
        temps = [1.0, 1.5, 2.0, 3.0, 5.0]

        # Initialize replicas with diverse starting points
        # Sort starts by function value and assign best to coldest replica
        evaluated_starts = []
        for s in starts[:n_explore]:
            x0 = np.clip(s, -5, 5)
            if calls >= 20 or time_left() < 5:
                break
            f0 = func(x0)
            calls += 1
            evaluated_starts.append((x0, f0))

        # Sort by function value
        evaluated_starts.sort(key=lambda t: t[1])

        # Initialize CMA-ES state for each replica
        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        c_sigma = (lam + 2) / (dim + lam + 5)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((lam / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + lam))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        replicas = []
        for i in range(n_replicas):
            if i < len(evaluated_starts):
                x_init, f_init = evaluated_starts[i]
            else:
                x_init = rng.uniform(-4, 4, dim)
                f_init = func(x_init)
                calls += 1
                if f_init < best_f:
                    best_f = f_init
                    best_x = x_init.copy()

            base_sigma = 1.5 * temps[i]
            replicas.append({
                'x': x_init.copy(),
                'sigma': base_sigma,
                'pc': np.zeros(dim),
                'ps': np.zeros(dim),
                'C': np.eye(dim),
                'B': np.eye(dim),
                'D': np.ones(dim),
                'f': f_init,
            })

        # Parallel Tempering main loop
        swap_interval = 5  # Swap every 5 generations
        gen = 0

        while calls < budget and time_left() > 10:
            gen += 1
            should_swap = (gen % swap_interval == 0)

            # Generate and evaluate offspring for each replica
            new_offspring = [None] * n_replicas
            for i in range(n_replicas):
                if calls >= budget or time_left() < 5:
                    break
                rep = replicas[i]
                x = rep['x']
                sigma = rep['sigma']
                B = rep['B']
                D = rep['D']

                offspring = []
                for j in range(lam):
                    if calls >= budget or time_left() < 2:
                        break
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    offspring.append((y, fy))
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(offspring) < mu:
                    new_offspring[i] = None
                    continue

                offspring.sort(key=lambda t: t[1])

                # Update best for this replica
                if offspring[0][1] < rep['f']:
                    rep['f'] = offspring[0][1]

                # Mean of selected
                x_new = sum(weights[j] * offspring[j][0] for j in range(mu))

                # Delta
                diff = (x_new - x) / (sigma + 1e-16)

                # Evolution path for sigma
                ps = (1 - c_sigma) * rep['ps'] + np.sqrt(c_sigma * (2 - c_sigma) * mueff / dim) * diff

                # Evolution path for covariance
                pc = (1 - c_c) * rep['pc'] + c_c * diff

                # Covariance matrix update
                H = np.zeros((dim, dim))
                for j in range(mu):
                    diff_j = (offspring[j][0] - x) / (sigma + 1e-16)
                    H += weights[j] * np.outer(diff_j, diff_j)

                C = (1 + c_1 + c_mu) * rep['C'] - c_1 * np.outer(pc, pc) - c_mu * H

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D_new = np.sqrt(eigvals)
                B_new = eigvecs

                # Step size adaptation
                sigma_new = sigma * np.exp(c_sigma / chi * (np.linalg.norm(ps) / chi - 1))
                sigma_new = max(sigma_new, 1e-6)

                new_offspring[i] = {
                    'x': x_new,
                    'sigma': sigma_new,
                    'pc': pc,
                    'ps': ps,
                    'C': C,
                    'B': B_new,
                    'D': D_new,
                    'f': rep['f'],
                }

            # Update replicas with new states
            for i in range(n_replicas):
                if new_offspring[i] is not None:
                    replicas[i] = new_offspring[i]

            # Parallel Tempering swap: best and worst replicas exchange x and sigma
            if should_swap and n_replicas >= 2:
                # Find best (lowest f) and worst (highest f) replicas
                f_vals = [rep['f'] for rep in replicas]
                best_idx = np.argmin(f_vals)
                worst_idx = np.argmax(f_vals)

                if best_idx != worst_idx:
                    # Swap x and sigma (and covariance structure) between best and worst
                    # This allows the cold replica to explore and the hot replica to exploit
                    rep_best = replicas[best_idx]
                    rep_worst = replicas[worst_idx]

                    # Swap positions and step sizes
                    temp_x = rep_best['x'].copy()
                    temp_sigma = rep_best['sigma']
                    temp_C = rep_best['C'].copy()
                    temp_B = rep_best['B'].copy()
                    temp_D = rep_best['D'].copy()
                    temp_pc = rep_best['pc'].copy()
                    temp_ps = rep_best['ps'].copy()

                    rep_best['x'] = rep_worst['x'].copy()
                    rep_best['sigma'] = rep_worst['sigma']
                    rep_best['C'] = rep_worst['C'].copy()
                    rep_best['B'] = rep_worst['B'].copy()
                    rep_best['D'] = rep_worst['D'].copy()
                    rep_best['pc'] = rep_worst['pc'].copy()
                    rep_best['ps'] = rep_worst['ps'].copy()

                    rep_worst['x'] = temp_x
                    rep_worst['sigma'] = temp_sigma
                    rep_worst['C'] = temp_C
                    rep_worst['B'] = temp_B
                    rep_worst['D'] = temp_D
                    rep_worst['pc'] = temp_pc
                    rep_worst['ps'] = temp_ps

        # Phase 3: Final local polish with Nelder-Mead if budget remains
        if calls < budget and time_left() > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f

            # Build initial simplex around best point
            simplex = [x.copy()]
            for d in range(dim):
                pt = x.copy()
                step = 0.1
                pt[d] += step
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            fvals = [fx]
            for pt in simplex[1:]:
                if calls >= budget or time_left() < 2:
                    break
                fv = func(pt)
                calls += 1
                fvals.append(fv)
                if fv < fx:
                    fx = fv
                    x = pt.copy()

            # Nelder-Mead iterations
            for _ in range(50):
                if calls >= budget or time_left() < 2:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls >= budget or time_left() < 2:
                    break
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls >= budget or time_left() < 2:
                        break
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls >= budget or time_left() < 2:
                        break
                    fc = func(xc)
                    calls += 1
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls >= budget or time_left() < 2:
                                break
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < fx:
                    fx = fvals[min_idx]
                    x = simplex[min_idx].copy()
                    best_x = x.copy()
                    best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
