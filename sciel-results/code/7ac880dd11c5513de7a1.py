"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f6 (Attractive sector) with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma0, max_evals, seed):
            """Run CMA-ES with given initial mean and sigma."""
            nonlocal best_x, best_f, call_count
            n = dim
            rng = np.random.default_rng(seed)

            # CMA-ES parameters
            mu = max(1, n // 2)  # number of selected parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (2 - cc)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

            # State
            mean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Evaluate initial population
            pop = []
            fvals = []
            for i in range(mu):
                z = rng.standard_normal(n)
                x = mean + sigma * (C @ z)
                fx = safe_call(x)
                if fx is None:
                    return call_count
                pop.append(x)
                fvals.append(fx)

            evals_used = mu
            pop = np.array(pop)
            fvals = np.array(fvals)

            while evals_used < max_evals:
                if time.time() > deadline:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update mean
                new_mean = np.sum(weights[:, None] * pop, axis=0)

                # Update path
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) / sigma * (new_mean - mean)
                cs_norm = np.linalg.norm(pc)
                hsig = cs_norm / np.sqrt(1 - (1 - cc) ** (evals_used // (mu + 1) + 1)) / np.sqrt(np.trace(C)) / n < 1.4 + 2 / (n - 1)

                ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) / sigma * (new_mean - mean)

                # Update covariance
                delta_d = (1 + hsig * (1 / damps - 1)) * (1 - (1 - cs) ** (evals_used // (mu + 1) + 1))
                rank1 = np.outer(ps, ps)
                rankmu = np.zeros((n, n))
                for i in range(mu):
                    zi = (pop[i] - mean) / sigma
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * delta_d - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

                # Symmetrize
                C = 0.5 * (C + C.T)

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))

                # Update mean
                mean = new_mean

                # Check for stagnation: if sigma is very small or no improvement
                if sigma < 1e-10 or np.all(fvals[:mu] > best_f - 1e-10):
                    # Try a small random perturbation to escape
                    if time.time() > deadline:
                        break
                    mean = mean + 0.1 * rng.standard_normal(n)
                    sigma = max(sigma, 0.5)
                    C = np.eye(n)
                    pc = np.zeros(n)
                    ps = np.zeros(n)

                # Generate next population
                pop = []
                fvals = []
                for i in range(mu):
                    z = rng.standard_normal(n)
                    x = mean + sigma * (C @ z)
                    fx = safe_call(x)
                    if fx is None:
                        return call_count
                    pop.append(x)
                    fvals.append(fx)
                    evals_used += 1

                pop = np.array(pop)
                fvals = np.array(fvals)

            return call_count

        # Multi-start CMA-ES
        # Phase 1: Quick random sampling to find good regions
        n_initial = min(20, budget // 10)
        for i in range(n_initial):
            if call_count >= budget or time.time() > deadline:
                break
            x = np.random.default_rng(100 + i).uniform(-4.5, 4.5, dim)
            safe_call(x)

        # Phase 2: Run CMA-ES from best found point
        remaining = budget - call_count
        if remaining > 10:
            cma_es(best_x.copy(), 2.0, remaining, seed=42)

        # Phase 3: If still have budget, try from a few other good points
        if call_count < budget and time.time() < deadline:
            # Try from center if not tried
            center = np.zeros(dim)
            if np.linalg.norm(center - best_x) > 1.0:
                safe_call(center)

        return best_x, best_f
# EVOLVE-BLOCK-END
