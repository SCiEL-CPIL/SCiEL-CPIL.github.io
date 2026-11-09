"""Improved BBOB f22 optimizer: CMA-ES + DE + Nelder-Mead + (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        rng = np.random.default_rng(42)
        dim = self.dim

        # Nelder-Mead implementation
        def nelder_mead(x0, f0, max_iters=100):
            nonlocal best_x, best_f, calls_made
            if not check_time() or (self.budget - calls_made) < 3:
                return x0, f0

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, -0.5, 0.5
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.1 if xi[i] < 4.9 else -0.1
                xi = np.clip(xi, -5, 5)
                fi = safe_func(xi)
                simplex.append(xi)
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            for it in range(max_iters):
                if not check_time() or (self.budget - calls_made) < dim + 2:
                    break
                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fr = safe_func(xr)
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fe = safe_func(xe)
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = np.clip(centroid + sigma_nm * (xr - centroid), -5, 5)
                    else:
                        # Inside contraction
                        xc = np.clip(centroid - sigma_nm * (centroid - simplex[-1]), -5, 5)
                    fc = safe_func(xc)
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                            fvals[i] = safe_func(simplex[i])
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                # Convergence check
                if np.std(fvals) < 1e-10:
                    break

            # Return best point in simplex
            best_idx = np.argmin(fvals)
            return simplex[best_idx].copy(), fvals[best_idx]

        # Differential Evolution
        def differential_evolution(x0, f0, max_gens=50, pop_size=20):
            nonlocal best_x, best_f, calls_made
            if not check_time() or (self.budget - calls_made) < pop_size:
                return

            CR, F = 0.7, 0.8
            # Initialize population
            pop = [x0.copy()]
            fpop = [f0]
            for i in range(pop_size - 1):
                if not check_time() or (self.budget - calls_made) < pop_size - i:
                    break
                xi = rng.uniform(-5, 5, dim)
                fi = safe_func(xi)
                pop.append(xi)
                fpop.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            for gen in range(max_gens):
                if not check_time() or (self.budget - calls_made) < pop_size:
                    break
                for i in range(len(pop)):
                    if not check_time() or (self.budget - calls_made) < pop_size - i:
                        break
                    # Select three distinct random individuals
                    r1, r2, r3 = rng.choice(len(pop), 3, replace=False)
                    while r1 == r2 or r2 == r3 or r1 == r3:
                        r1, r2, r3 = rng.choice(len(pop), 3, replace=False)

                    # Mutation
                    v = np.clip(pop[r1] + F * (pop[r2] - pop[r3]), -5, 5)
                    # Crossover
                    u = rng.integers(0, dim)
                    trial = np.copy(pop[i])
                    for j in range(dim):
                        if rng.random() < CR or j == u:
                            trial[j] = v[j]
                    trial = np.clip(trial, -5, 5)
                    ft = safe_func(trial)
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                    if ft <= fpop[i]:
                        pop[i] = trial
                        fpop[i] = ft

        # CMA-ES implementation (optimized)
        def cma_es(x0, f0, budget_remaining, max_steps):
            nonlocal best_x, best_f, calls_made

            if budget_remaining <= 0 or not check_time():
                return

            # Population size
            lam = max(4, int(4 + 3 * np.log(dim)))
            # Step size
            sigma = 0.5
            # Covariance matrix
            C = np.eye(dim)
            # Evolution path
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Step size control parameters
            cs = (lam // 2 + 1) / (dim + lam // 2 + 5)
            ds = 1 + 2 * max(0, (dim - 1) / (dim + 2) - 1) + cs
            c1 = (2 + 2 / (dim + 1.7)) / ((dim + 2) ** 2 + lam)
            cmu = min(2 / ((dim + 1.3) ** 2 + lam), (1 - c1) * 0.5)
            cc = (4 + dim / lam) / (dim + 4 + 2 * lam / dim)

            # Damping for cumulative covariance matrix update
            mu = lam // 2
            weights = np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights ** 2)

            x_mean = x0.copy()

            for step in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= lam:
                    break

                # Precompute Cholesky once per generation
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = (C + C.T) / 2
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-12:
                        C += (1e-12 - np.min(eigvals) + 1e-9) * np.eye(dim)
                    L = np.linalg.cholesky(C)

                # Generate population
                offspring = []
                for i in range(lam):
                    if not check_time() or (self.budget - calls_made) <= lam - i:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_func(y)
                    offspring.append((fy, y))

                if len(offspring) == 0:
                    break

                # Sort offspring
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()

                # Select best mu offspring
                mu_actual = min(mu, len(offspring))
                y_w = np.zeros(dim)
                for i in range(mu_actual):
                    y_w += weights[i] * offspring[i][1]
                y_w /= np.sum(weights[:mu_actual])

                # Update mean
                x_mean = y_w

                # Update step size
                # Compute evolution path
                try:
                    C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(dim)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (y_w - x_mean)) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (step + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))

                # Update cumulative covariance matrix evolution path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (y_w - x_mean) / sigma

                # Update covariance matrix
                delta_d = (1 + 2 * (hsig - 1) * cc * (2 - cc)) * c1
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu_actual):
                    zi = (offspring[i][1] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = (1 + c1 * (delta_d - 1) + cmu * (mueff * (c1 + cmu) / c1 - 1)) * C + cmu * rank_mu - c1 * rank_one

                # Ensure positive definiteness
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += (1e-12 - np.min(eigvals) + 1e-9) * np.eye(dim)

                # Update step size
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt((dim + 1) / (dim - 1)) - 1))

                # Clamp sigma
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Convergence check
                if sigma < 1e-10 or np.max(np.abs(C)) < 1e-12:
                    break

        # Phase 1: Quick exploration with random sampling
        n_random = min(50, max(10, self.budget // 10))
        random_points = []
        for i in range(n_random):
            if not check_time() or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            random_points.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort random points by fitness
        random_points.sort(key=lambda t: t[0])

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES from the best random point
        if check_time() and (self.budget - calls_made) > 20:
            budget_for_cma = self.budget - calls_made
            max_steps = max(1, budget_for_cma // max(4, int(4 + 3 * np.log(dim))))
            cma_es(best_x, best_f, budget_for_cma, max_steps)

        # Phase 3: DE from the best point
        remaining = self.budget - calls_made
        if remaining > 20 and check_time():
            differential_evolution(best_x, best_f, max_gens=30, pop_size=min(20, remaining // 2))

        # Phase 4: Nelder-Mead local search
        remaining = self.budget - calls_made
        if remaining > 10 and check_time():
            best_x, best_f = nelder_mead(best_x, best_f, max_iters=min(50, remaining // (dim + 2)))

        # Phase 5: More CMA-ES restarts from other good points
        remaining = self.budget - calls_made
        if remaining > 30 and check_time():
            for idx in range(1, min(3, len(random_points))):
                if not check_time() or (self.budget - calls_made) < 20:
                    break
                x_start = random_points[idx][1].copy()
                f_start = random_points[idx][0]
                budget_for_cma = self.budget - calls_made
                max_steps = max(1, budget_for_cma // max(4, int(4 + 3 * np.log(dim))))
                cma_es(x_start, f_start, budget_for_cma, max_steps)

        # Phase 6: Final Nelder-Mead polish
        remaining = self.budget - calls_made
        if remaining > 10 and check_time():
            best_x, best_f = nelder_mead(best_x, best_f, max_iters=min(30, remaining // (dim + 2)))

        # Phase 7: Adaptive (1+1)-ES local search with 1/5 success rule
        remaining = self.budget - calls_made
        if remaining > 0 and check_time():
            sigma_local = 0.3
            x = best_x.copy()
            fx = best_f
            for _ in range(min(remaining, 150)):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.4
                else:
                    sigma_local *= 0.85
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
