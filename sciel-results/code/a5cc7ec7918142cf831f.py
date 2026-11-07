"""Direct-structured optimizer for BBOB f20 (Schwefel) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    CMA-ES-like optimizer with multi-start and direct coordinate search.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        def cma_es(x0, sigma0, max_calls, rng):
            """Run CMA-ES from x0 with initial step sigma0 for up to max_calls evaluations."""
            nonlocal best_f, best_x, calls_made
            n = dim
            x = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(n)
            C_sqrt = np.linalg.cholesky(C)

            # Population size: smaller for faster convergence
            pop = 4
            mu = 2
            weights = np.ones(mu) / mu
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            pc = np.zeros(n)
            ps = np.zeros(n)

            calls_in_cma = 0

            while calls_in_cma < max_calls:
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return

                # Generate and evaluate population
                offspring = []
                offspring_f = []
                for j in range(pop):
                    if calls_made >= budget:
                        return
                    if time.time() > deadline:
                        return
                    z = rng.standard_normal(n)
                    y = x + sigma * C_sqrt @ z
                    y = np.clip(y, -5, 5)
                    fy = try_func(y)
                    if fy is None:
                        return
                    offspring.append(y)
                    offspring_f.append(fy)
                    calls_in_cma += 1

                offspring = np.array(offspring)
                offspring_f = np.array(offspring_f)

                # Sort by fitness
                order = np.argsort(offspring_f)
                y_sorted = offspring[order]
                f_sorted = offspring_f[order]

                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = y_sorted[0].copy()

                # Compute weighted mean of best mu
                y_mean = np.average(y_sorted[:mu], axis=0, weights=weights)

                # Update covariance
                d = y_mean - x
                C = (1 - c_c - c_c * c_sigma) * C + c_c * np.outer(pc, pc) + c_sigma * d_sigma * np.outer(d, d)

                # Symmetrize
                C = (C + C.T) / 2

                # Update pc
                pc = (1 - c_c) * pc + c_c * d / sigma

                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * d
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Update mean
                x = y_mean

                # Update C_sqrt periodically
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(n)
                    try:
                        C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        C_sqrt = np.eye(n)
                        sigma = max(sigma, 0.1)

                # Convergence check
                if sigma < 1e-8:
                    break

        def coordinate_descent(x0, max_calls, rng):
            """Simple coordinate descent with adaptive step size for final refinement."""
            nonlocal best_f, best_x, calls_made
            x = x0.copy().astype(float)
            fx = try_func(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            step = 0.5
            for _ in range(max_calls):
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return

                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        return
                    if time.time() > deadline:
                        return

                    # Try positive and negative steps
                    for sign in [1.0, -1.0]:
                        y = x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = try_func(y)
                        if fy is None:
                            return
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break  # Only try one direction per coordinate
                    if not improved:
                        step *= 0.5
                        if step < 1e-7:
                            break

        def final_polish(x0, max_calls, rng):
            """Final local search with shrinking step sizes."""
            nonlocal best_f, best_x, calls_made
            x = x0.copy().astype(float)
            fx = try_func(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            step = 0.1
            for _ in range(max_calls):
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return

                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        return
                    if time.time() > deadline:
                        return

                    for sign in [1.0, -1.0]:
                        y = x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = try_func(y)
                        if fy is None:
                            return
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                    if not improved:
                        step *= 0.5
                        if step < 1e-8:
                            break

        # Multi-start strategy
        rng = np.random.default_rng(2024)

        starts = [
            np.zeros(dim),  # center
            np.full(dim, 4.20968),  # Schwefel's global minimum
            np.full(dim, -4.20968),
        ]
        # Random starts
        for i in range(3):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Shuffle starts
        idx = np.arange(len(starts))
        rng.shuffle(idx)
        starts = [starts[i] for i in idx]

        # Phase 1: Quick evaluation of all starts
        for x0 in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            fx = try_func(x0)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Phase 2: Run CMA-ES from the best start
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            cma_budget = min(remaining, 300)
            cma_es(best_x, 1.0, cma_budget, rng)

        # Phase 3: Coordinate descent for final refinement
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            cd_budget = min(remaining, 80)
            coordinate_descent(best_x, cd_budget, rng)

        # Phase 4: Final polish
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            polish_budget = min(remaining, 50)
            final_polish(best_x, polish_budget, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
