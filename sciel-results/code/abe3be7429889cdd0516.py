import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if time.time() > deadline:
                raise TimeoutError()
            if calls >= budget:
                raise BudgetExceeded()
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        class BudgetExceeded(Exception):
            pass

        # --- Phase 1: Latin Hypercube Sampling for initial exploration ---
        # Use a diverse set of points to cover the search space
        n_lhc = 30
        lhc_points = np.zeros((n_lhc, dim))
        for i in range(dim):
            # Latin hypercube: divide each dimension into n_lhc equal intervals
            # and pick one random point from each interval
            intervals = np.linspace(-5, 5, n_lhc + 1)
            random_points = np.array([rng.uniform(intervals[j], intervals[j+1]) for j in range(n_lhc)])
            # Permute to ensure one point per interval per dimension
            perm = rng.permutation(n_lhc)
            lhc_points[:, i] = random_points[perm]

        for i in range(n_lhc):
            try:
                evaluate(lhc_points[i])
            except (BudgetExceeded, TimeoutError):
                break

        # --- Phase 2: CMA-ES for exploitation ---
        # Initialize CMA-ES with the best point found so far
        x = best_x.copy()
        sigma = 1.0  # Initial step size
        popsize = 10 + int(3 * np.log(dim))  # For dim=5, popsize=17
        mu = popsize // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # CMA-ES parameters
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        dampc = (4 * mu_eff - 2) / (dim + 4)

        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        DC = np.linalg.cholesky(C)
        C_inv_half = np.linalg.inv(DC).T @ np.linalg.inv(DC)

        # CMA-ES loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = rng.standard_normal((popsize, dim))
            zs = np.array([DC @ y for y in ys])
            xs = np.array([x + sigma * z for z in zs])
            xs = np.clip(xs, -5, 5)

            # Evaluate population
            fs = np.zeros(popsize)
            for i in range(popsize):
                try:
                    fs[i] = evaluate(xs[i])
                except (BudgetExceeded, TimeoutError):
                    break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            xs_sorted = xs[order]

            # Check time again
            if time.time() > deadline:
                break

            # Weighted mean
            x_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)

            # Update evolution paths
            diff = x_new - x
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C_inv_half @ diff)

            # Update pc
            norm_ps = np.linalg.norm(ps)
            threshold = (1.4 + 2 / (dim + 1)) * np.sqrt(
                (1 - (1 - cs) ** 2) / (1 - (1 - cs) ** (2 * calls / (dim * damps)))
            )
            if norm_ps / np.sqrt(1 - (1 - cs) ** (2 * calls / (dim * damps))) < threshold:
                pc = (1 - c1) * pc + c1 / sigma * diff

            # Update covariance matrix
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
            C += cmu * np.sum(w[:, None, None] * np.outer(ys_sorted[:mu], ys_sorted[:mu]) * w[:, None], axis=0)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                DC = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                C_inv_half = np.linalg.inv(DC).T @ np.linalg.inv(DC)
            except np.linalg.LinAlgError:
                DC = np.eye(dim)
                C_inv_half = np.eye(dim)
                C = np.eye(dim)

            # Update mean
            x = x_new

            # Adapt sigma using 1/5 rule
            successes = np.sum(fs_sorted[:mu] < fs_sorted[mu] if mu < popsize else fs_sorted[:mu] < np.median(fs))
            if successes > mu * 0.5:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_f
