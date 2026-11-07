"""Pure CMA-ES with multi-restart for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline - 2.0 and calls_used < budget

        def safe_call(x):
            nonlocal calls_used
            if not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            try:
                fx = func(x)
                fx = float(fx)
            except Exception:
                return None
            calls_used += 1
            return x.copy(), fx

        # Phase 1: Initial random exploration to find a good starting point
        n_init = min(30, budget)
        candidates = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            result = safe_call(x)
            if result is None:
                break
            x_c, fx = result
            candidates.append((fx, x_c))

        if not candidates:
            result = safe_call(np.zeros(dim))
            if result:
                return result[0], result[1]
            return np.zeros(dim), 0.0

        candidates.sort(key=lambda t: t[0])
        best_x = candidates[0][1].copy()
        best_fx = candidates[0][0]

        # Phase 2: CMA-ES with restarts
        # We'll do multiple CMA-ES runs, each with a portion of the remaining budget
        
        remaining = budget - calls_used
        n_restarts = 3
        budget_per_restart = max(20, remaining // n_restarts)

        for restart in range(n_restarts):
            if not time_ok():
                break
            
            # For restart 0, start from the best point found
            # For subsequent restarts, start from a perturbed version of the best
            if restart == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            else:
                # Perturb the best point to escape local minima
                x0 = best_x + rng.normal(0, 1.0, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.5

            # CMA-ES parameters
            lambda_pop = max(12, dim + 3 + int(2 * np.log(dim)))
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs

            c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma)
            cmu = min(1 - c1, 2 * (weights[0] - 0.5)) * c_sigma * c_sigma

            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            C = np.eye(dim)
            csqrt = np.eye(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x = x0.copy()
            sigma = sigma0
            no_improve = 0
            best_local_fx = best_fx
            best_local_x = best_x.copy()
            start_calls = calls_used
            max_gen = budget_per_restart // lambda_pop

            for gen in range(max_gen):
                if not time_ok() or (calls_used - start_calls) >= budget_per_restart:
                    break

                # Generate population
                pop = []
                for i in range(lambda_pop):
                    if not time_ok() or (calls_used - start_calls) >= budget_per_restart:
                        break
                    zi = rng.standard_normal(dim)
                    xi = x + sigma * (csqrt @ zi)
                    result = safe_call(xi)
                    if result is None:
                        break
                    xi_c, fi = result
                    pop.append((fi, xi_c))

                if len(pop) < mu:
                    break

                pop.sort(key=lambda t: t[0])

                # Track best
                if pop[0][0] < best_local_fx:
                    best_local_fx = pop[0][0]
                    best_local_x = pop[0][1].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Check for stagnation
                if no_improve > 20:
                    break

                # Update CMA-ES state
                x_new = np.sum([weights[i] * pop[i][1] for i in range(min(mu, len(pop)))], axis=0)

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (csqrt.T @ (x_new - x))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < \
                       (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x_new - x)

                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(pop))):
                    y = (pop[i][1] - x) / sigma
                    rankmu += weights[i] * np.outer(y, y)

                C = (1 + c1 + cmu - c1 - c_sigma) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2

                # Ensure positive definiteness
                try:
                    w, V = np.linalg.eigh(C)
                    w = np.maximum(w, 1e-14)
                    C = V @ np.diag(w) @ V.T
                    csqrt = V @ np.diag(np.sqrt(w)) @ V.T
                except Exception:
                    csqrt = np.eye(dim)
                    C = np.eye(dim)

                # Update sigma
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 0.001, 10.0)
                x = x_new

            # Update global best
            if best_local_fx < best_fx:
                best_fx = best_local_fx
                best_x = best_local_x.copy()

        # Phase 3: Final local refinement with a simple coordinate-wise search
        remaining = budget - calls_used
        if remaining > 10 and time_ok():
            # Try small perturbations in each coordinate
            step_sizes = [0.5, 0.1, 0.01, 0.001]
            for step in step_sizes:
                if not time_ok():
                    break
                improved = True
                while improved and time_ok():
                    improved = False
                    for d in range(dim):
                        for direction in [-1, 1]:
                            x_try = best_x.copy()
                            x_try[d] += direction * step
                            x_try = np.clip(x_try, -5, 5)
                            result = safe_call(x_try)
                            if result is None:
                                break
                            x_c, fx = result
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x_c.copy()
                                improved = True

        return best_x, best_fx
# EVOLVE-BLOCK-END
