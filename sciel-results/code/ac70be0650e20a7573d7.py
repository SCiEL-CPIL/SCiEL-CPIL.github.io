"""CMA-ES optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    - Maintains a multivariate Gaussian distribution over the search space.
    - Adapts mean, step size, and covariance matrix based on selected solutions.
    - Exploits anisotropic structure of f14 through covariance adaptation.
    - Deadline guard to ensure we return before the time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(d))
        lam = 4 + int(3 * np.log(d))
        mu = lam // 2  # number of selected solutions

        # Weights for selected solutions
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        cs = (mu_eff + 2) / (d + mu_eff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Step size control
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initial parameters
        x_mean = np.zeros(d)
        sigma = 2.0  # initial step size
        C = np.eye(d)  # covariance matrix
        pc = np.zeros(d)  # evolution path for covariance
        ps = np.zeros(d)  # evolution path for step size

        # Track evaluations
        evals = []

        def record_eval(x, fx):
            if fx is not None:
                evals.append((x.copy(), fx))

        # Helper: sample from multivariate normal with given mean, sigma, and C
        def sample_mvnormal(mean, sigma, C, n=1):
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                L = np.linalg.cholesky(L + 1e-10 * np.eye(d))

            z = rng.standard_normal((n, d))
            return mean + sigma * (z @ L.T)

        # Main CMA-ES loop
        while calls_used < max_calls and time.time() < deadline - 10:
            # Generate population
            pop = sample_mvnormal(x_mean, sigma, C, lam)
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            pop_f = []
            for x in pop:
                fx = try_eval(x)
                if fx is None:
                    break
                pop_f.append(fx)
                record_eval(x, fx)

            if len(pop_f) < lam:
                # Not enough evaluations, break
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_sorted = pop[indices]
            pop_f_sorted = [pop_f[i] for i in indices]

            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum(w[:, None] * pop_sorted[:mu], axis=0)

            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C @ (x_mean - old_mean)) / sigma

            # Hensel checkpoint: if ||ps|| > c * chi_n, adapt sigma
            if np.linalg.norm(ps) > 1.4 * chi_n:
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                ps = ps / np.linalg.norm(ps) * 1.4 * chi_n

            # Update evolution path for covariance
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (x_mean - old_mean) / sigma

            # Update covariance matrix
            delta_c = (1 + 2 * max(0, (np.linalg.norm(pc) / chi_n - 1) / (d + 1))) * cc
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                zi = (pop_sorted[i] - old_mean) / sigma
                rank_mu += w[i] * np.outer(zi, zi)

            C = (1 + c1 * (delta_c - 1) - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Check termination conditions
            # 1. Small step size
            if sigma < 1e-12:
                break
            # 2. Small spread in population
            if np.max(C) - np.min(C) < 1e-16:
                break
            # 3. No improvement in last many generations
            if len(evals) > 50:
                recent = [f for _, f in evals[-50:]]
                if max(recent) - min(recent) < 1e-10:
                    break

        # Final polish: use remaining budget for fine-tuning around best_x
        if calls_used < max_calls and time.time() < deadline - 2 and best_x is not None:
            polish_center = best_x.copy()
            polish_sigma = max(0.01, sigma * 0.1)
            polish_C = np.eye(d)
            polish_no_improve = 0
            max_polish_no_improve = 8

            while calls_used < max_calls and time.time() < deadline - 2:
                # Sample a few candidates around best
                candidates = sample_mvnormal(polish_center, polish_sigma, polish_C, 3)
                candidates = np.clip(candidates, -5, 5)

                improved = False
                for x in candidates:
                    fx = try_eval(x)
                    if fx is None:
                        break
                    record_eval(x, fx)
                    if fx < best_f - 1e-12:
                        best_f = fx
                        best_x = x.copy()
                        polish_center = x.copy()
                        improved = True

                if improved:
                    polish_no_improve = 0
                    polish_sigma = min(polish_sigma * 1.2, 0.5)
                else:
                    polish_no_improve += 1
                    polish_sigma = max(polish_sigma * 0.7, 0.001)

                    if polish_no_improve >= max_polish_no_improve:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
