"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                x = np.asarray(x, dtype=np.float64)
                fx = func(x)
                fx = float(fx)
                if np.isfinite(fx) and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        def cma_es_run(x0, sigma0, max_calls):
            nonlocal call_count, best_x, best_f
            n = dim
            if call_count >= budget:
                return

            # CMA-ES parameters
            mu = max(1, n // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            calls_used = 0

            while calls_used < max_calls and call_count < budget:
                if time.time() > deadline - 1.0:
                    break

                # Sample lambda points
                lam = max(4, 4 + int(3 * n))
                # Ensure we don't exceed budget
                if lam > budget - call_count:
                    lam = max(1, budget - call_count)
                if lam < 2:
                    break

                # Generate samples
                ys = []
                for i in range(lam):
                    if call_count + i >= budget:
                        break
                    y = rng.standard_normal(n)
                    ys.append(y)
                lam = len(ys)
                if lam < 2:
                    break

                # Evaluate
                xs = np.array([x + sigma * y for y in ys])
                fs = np.array([safe_call(xi) for xi in xs])
                calls_used += lam

                # Sort by fitness
                idx = np.argsort(fs)
                ys = np.array(ys)[idx]
                fs = fs[idx]

                # Update mean
                x = x + sigma * np.sum(w[:mu] * ys[:mu], axis=0)

                # Update step size control
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C @ (np.sum(w[:mu] * ys[:mu], axis=0) * sigma)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance matrix
                ys_w = np.array([w[i] * (ys[i] - np.sum(w[:mu] * ys[:mu], axis=0)) for i in range(mu)])
                delta_sig = np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1)) - 1
                C = (1 + c_1 * c_mu + c_1 * c_sigma * delta_sig * (1 - c_sigma) - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) - C)
                C += c_mu * np.sum(ys_w * ys_w.T, axis=0)

                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (np.sum(w[:mu] * ys[:mu], axis=0) * sigma)

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2

                # Check convergence
                if sigma < 1e-12 or np.max(np.abs(x)) > 1e6:
                    break

                # Check if we've made enough progress to stop early
                if calls_used > 100 and np.std(fs[:mu]) < 1e-8:
                    break

            return x, fs[0] if len(fs) > 0 else np.inf

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Multi-start: try several initial points
        # First, try center
        starts = [np.zeros(dim)]
        # Add random starts
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, dim))

        # Distribute budget among starts
        num_starts = len(starts)
        # Give more budget to earlier starts
        weights = np.array([1.0 / (i + 1) for i in range(num_starts)])
        weights = weights / weights.sum()
        budgets = (weights * budget).astype(int)
        # Ensure at least 10 calls per start
        for i in range(num_starts):
            if budgets[i] < 10:
                budgets[i] = 10
        # Normalize to not exceed budget
        total = budgets.sum()
        if total > budget:
            scale = budget / total
            budgets = (budgets * scale).astype(int)
            # Fix rounding
            diff = budget - budgets.sum()
            budgets[0] += diff

        for i, x0 in enumerate(starts):
            if call_count >= budget:
                break
            if time.time() > deadline - 1.0:
                break
            max_calls = budgets[i]
            cma_es_run(x0, 2.0, max_calls)

        return best_x, best_f
# EVOLVE-BLOCK-END
