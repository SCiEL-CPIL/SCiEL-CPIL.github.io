"""CMA-ES with adaptive restarts and coordinate scan for Katsuura."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB-style Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        best_x = None
        best_f = np.inf
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 1.0:
                raise RuntimeError("Time limit")
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        def cma_es(x0, sigma0, max_calls):
            nonlocal calls_made, best_f, best_x, deadline
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            # CMA-ES parameters
            cc = (4 + n / lam) / ((n + 4) + 4 / lam)
            cs = (n + 1.3) / (n + 3 + 2 * n / lam)
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = int(np.sum(weights[:int(1.0 / weights[0])], axis=0))
            c1 = (2 * (mueff - 2 + 2.0 / n)) / ((n + 2) ** 2 + mueff - 2 + 2.0 / n)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 2) ** 2 + mueff - 2 + 1.0 / mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1) - 1) - 1) + cs
            sigma = sigma0
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            # Precompute matrix square root
            D, V = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            sqrtC = V @ np.diag(D) @ V.T
            # Initial mean
            mean = np.array(x0, dtype=float, copy=True)
            mean = np.clip(mean, -5, 5)

            while calls_made < budget and time.time() < deadline - 3.0:
                # Generate population
                offspring = []
                for i in range(lam):
                    z = np.random.standard_normal(n)
                    x = mean + sigma * (sqrtC @ z)
                    x = np.clip(x, -5, 5)
                    offspring.append(x)

                # Evaluate
                fvals = []
                for x in offspring:
                    try:
                        fvals.append(counted_func(x))
                    except RuntimeError:
                        return

                if len(fvals) < lam:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                offspring = [offspring[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update mean (weighted)
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * offspring[i]
                mean = new_mean
                mean = np.clip(mean, -5, 5)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (sqrtC.T @ ((mean - x0) / sigma if False else (np.mean([offspring[i] for i in range(mu)], axis=0) - mean) / sigma))
                # Actually, the standard formula uses the weighted mean difference
                # Let me redo: the weighted mean of the selected points
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (offspring[i] - mean)
                y_w /= sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_w
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lam + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
                # Update covariance
                delta_h = (2 * hsig - 1) * cc * (2 - cc)
                C = (1 + c1 * delta_h - c1 - cmu) * C
                for i in range(mu):
                    y_i = (offspring[i] - mean) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Recompute sqrt(C) periodically
                if calls_made % (lam * 5) < lam:
                    D, V = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-12))
                    sqrtC = V @ np.diag(D) @ V.T

                # Stagnation check: if best hasn't improved in several generations, break
                if len(fvals) > 0 and fvals[0] >= best_f - 1e-12:
                    # Check if we've been stuck
                    pass  # We'll handle restarts externally

                # Convergence check
                if sigma < 1e-10 or np.all(np.abs(mean) < 1e-10):
                    break

        # Phase 1: Coarse coordinate scan to find a good basin
        # Scan each dimension at a few points to get a rough idea
        n_scan_per_dim = 7
        scan_points = np.linspace(-4.5, 4.5, n_scan_per_dim)
        # Start from origin, scan one dimension at a time
        current = np.zeros(dim)
        for d in range(dim):
            for sp in scan_points:
                if calls_made >= budget or time.time() > deadline - 10:
                    break
                test_x = np.array(current, copy=True)
                test_x[d] = sp
                try:
                    counted_func(test_x)
                except RuntimeError:
                    break
            # Pick best value for this dimension
            # We need to re-evaluate to find which was best
            # Actually, let's just do a simpler approach: random samples first

        # Phase 1b: Random sampling to find good starting points
        n_random = 20
        rng = np.random.default_rng(123)
        random_candidates = []
        for i in range(n_random):
            if calls_made >= budget or time.time() > deadline - 10:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            try:
                f = counted_func(x)
                random_candidates.append((f, x))
            except RuntimeError:
                break

        # Sort by fitness
        random_candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from the best random point
        if random_candidates:
            best_start = random_candidates[0][1]
            cma_es(best_start, 1.0, budget)

        # Phase 3: Adaptive restarts if we haven't converged
        if calls_made < budget and time.time() < deadline - 10:
            n_restarts = 3
            for r in range(n_restarts):
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                # Perturb the best point
                if best_x is not None:
                    perturbation = rng.normal(0, 0.5, dim)
                    new_start = np.clip(best_x + perturbation, -4.5, 4.5)
                else:
                    new_start = rng.uniform(-4.5, 4.5, dim)
                cma_es(new_start, 0.5, budget)

        # Phase 4: Final local polish with coordinate descent
        if best_x is not None and calls_made < budget and time.time() < deadline - 3:
            # Coordinate descent
            for iteration in range(20):
                if calls_made >= budget or time.time() > deadline - 2:
                    break
                improved = False
                for d in range(dim):
                    # Try small steps in each direction
                    for step in [0.1, -0.1, 0.05, -0.05, 0.01, -0.01]:
                        if calls_made >= budget or time.time() > deadline - 1:
                            break
                        test_x = np.array(best_x, copy=True)
                        test_x[d] += step
                        test_x = np.clip(test_x, -5, 5)
                        try:
                            f = counted_func(test_x)
                            if f < best_f - 1e-12:
                                best_x = test_x
                                best_f = f
                                improved = True
                        except RuntimeError:
                            break
                if not improved:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
