"""CMA-ES optimizer for shifted/rotated Rosenbrock f8."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for Rosenbrock-style f8 in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            if call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x_arr = np.asarray(x, dtype=float)
            if x_arr.shape != (self.dim,):
                x_arr = x_arr.reshape(-1)[:self.dim]
            x_clipped = np.clip(x_arr, -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count < self.budget

        # --- Phase 1: Multi-start CMA-ES ---
        rng = np.random.default_rng(12345)

        # Define several starting points spread across the domain
        # Include origin, ones, and random points
        starts = []
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim))
        starts.append(np.full(self.dim, 2.0))
        starts.append(np.full(self.dim, -1.0))
        for _ in range(8):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))

        # Evaluate starting points first
        for s in starts:
            if not (budget_ok() and time_ok()):
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            wrapped(s_clipped)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped(best_x)

        # Run CMA-ES from the best starting point
        if budget_ok() and time_ok():
            self._run_cmaes(wrapped, best_x.copy(), best_f, rng, deadline)

        # If we have budget left, try a second CMA-ES from a different random start
        if budget_ok() and time_ok() and call_count < self.budget - 50:
            # Pick a new random start far from current best
            for attempt in range(5):
                if not (budget_ok() and time_ok()):
                    break
                new_start = rng.uniform(-4.5, 4.5, self.dim)
                # Check if it's far enough from current best
                if np.linalg.norm(new_start - best_x) > 2.0:
                    f_new = wrapped(new_start)
                    if f_new < best_f * 0.9 or f_new < best_f - 1.0:
                        self._run_cmaes(wrapped, new_start.copy(), f_new, rng, deadline)
                    break

        # --- Phase 2: Local refinement with small-step CMA-ES ---
        if budget_ok() and time_ok() and call_count < self.budget - 10:
            self._run_cmaes(wrapped, best_x.copy(), best_f, rng, deadline,
                            initial_sigma=0.05, max_iter=30)

        # --- Phase 3: Final micro-polish ---
        if budget_ok() and time_ok():
            x_es = best_x.copy()
            fx_es = best_f
            sigma = 0.01
            remaining = self.budget - call_count
            for _ in range(min(remaining, 30)):
                if not (budget_ok() and time_ok()):
                    break
                y = np.clip(x_es + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped(y)
                if fy < fx_es:
                    x_es, fx_es = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.92
                if fx_es < best_f:
                    best_f = fx_es
                    best_x = x_es.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f

    def _run_cmaes(self, func, x0, f0, rng, deadline, initial_sigma=None, max_iter=100):
        """Run a single CMA-ES instance."""
        n = self.dim
        if initial_sigma is None:
            initial_sigma = 1.0

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_ // 2  # number of selected solutions
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # Step size control
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # Chi-squared approximation
        chi_n = n * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x_mean = x0.copy()
        sigma = initial_sigma
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        evals = 0

        # Helper to update C, C_inv, C_sqrt, C_sqrt_inv
        def update_matrices(C, C_inv, C_sqrt, C_sqrt_inv, c_1, c_mu, w, x_mean, sigma, p_sigma, d_sigma, n, mueff, chi_n, gen):
            nonlocal C, C_inv, C_sqrt, C_sqrt_inv
            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (x_mean - x_mean)  # simplified
            # Actually need to compute properly from population
            # This is getting complex; let's use a simpler update

            # For simplicity in this implementation, we'll use a basic update
            # The full CMA-ES is complex; let's implement a working version

        # Simplified CMA-ES loop
        for gen in range(max_iter):
            if time.time() > deadline or evals >= self.budget:
                break

            # Generate population
            population = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = C_sqrt @ z
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)
                population.append(x_new)

            # Evaluate population
            fvals = []
            for x_new in population:
                if time.time() > deadline or evals >= self.budget:
                    break
                fvals.append(func(x_new))
                evals += 1

            if len(fvals) < lambda_:
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            selected = [population[i] for i in indices[:mu]]
            fvals_sorted = [fvals[i] for i in indices[:mu]]

            # Update mean
            x_new_mean = np.sum([w[i] * selected[i] for i in range(mu)], axis=0)

            # Update step size
            if sigma > 1e-12:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (x_new_mean - x_mean)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = min(sigma, 1e6)  # cap sigma
            else:
                sigma = 1e-6

            # Update covariance matrix
            if np.linalg.norm(x_new_mean - x_mean) > 1e-12:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_new_mean - x_mean)

            # Compute rank-1 and rank-mu updates
            delta_sig = (np.linalg.norm(p_sigma) / chi_n - 1) * c_sigma * d_sigma
            h_sig = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) <
                          (1.4 + 2 / (n + 1)) * chi_n) else 0

            # Rank-1 update
            C = (1 + c_1 * delta_sig * h_sig) * C
            C += c_1 * np.outer(p_c, p_c)
            if h_sig:
                C -= c_1 * (1 - c_c) * np.eye(n)  # approximation

            # Rank-mu update
            y_w = [(selected[i] - x_mean) / sigma for i in range(mu)]
            rank_mu = np.sum([w[i] ** 2 * np.outer(y_w[i], y_w[i]) for i in range(mu)], axis=0)
            C += c_mu * rank_mu

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C_inv and C_sqrt
            C_inv = np.linalg.inv(C)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

            x_mean = x_new_mean

            # Check for convergence
            if np.linalg.norm(x_new_mean - x_mean) < 1e-10 and sigma < 1e-6:
                break

        return x_mean, sigma
# EVOLVE-BLOCK-END
