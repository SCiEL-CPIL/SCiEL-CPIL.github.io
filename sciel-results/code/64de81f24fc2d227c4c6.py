"""Portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer: random sampling + CMA-ES-like with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Phase 1: Broad random sampling to find a good starting region
        # Use stratified sampling for better coverage
        n_samples = min(50, budget // 10)
        best_x = None
        best_f = np.inf

        # Latin Hypercube-style sampling
        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if time.time() > deadline:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: CMA-ES-like optimization with restarts
        # We'll do several rounds of local optimization from different starting points
        # Use a simplified CMA-ES with adaptive step size

        def cma_es(x0, f0, n_iter, sigma0, rng):
            """Simple CMA-ES variant."""
            x = x0.copy()
            f = f0
            sigma = sigma0
            n = len(x)
            # Covariance matrix (start with identity)
            C = np.eye(n)
            # Step sizes for each dimension
            step = np.ones(n) * sigma

            for i in range(n_iter):
                if time.time() > deadline:
                    break

                # Generate candidate
                # Use diagonal adaptation (simpler than full CMA-ES)
                y = x + step * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)

                if fy < f:
                    # Success: adapt step sizes
                    x, f = y, fy
                    # Increase step sizes for dimensions that improved
                    improvement = (x - x0)  # not quite right, let's use direction
                    step = np.maximum(step * 1.1, 0.01)
                    step = np.minimum(step, 5.0)
                else:
                    # Failure: decrease step sizes
                    step = step * 0.8
                    step = np.maximum(step, 0.001)

            return x, f

        # Multiple restarts with different strategies
        n_restarts = 5
        per_restart_budget = (budget - n_samples) // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break

            if r == 0:
                # First restart: continue from current best
                start_x = best_x.copy()
                start_f = best_f
                sigma0 = 1.0
            else:
                # Subsequent restarts: sample new starting points
                # Mix of random and perturbed best
                if r % 2 == 1:
                    start_x = best_x + rng.normal(0, 0.5, dim)
                    start_x = np.clip(start_x, -5, 5)
                else:
                    start_x = rng.uniform(-5, 5, dim)

                start_f = func(start_x)
                if start_f < best_f:
                    best_f = start_f
                    best_x = start_x.copy()

                sigma0 = 0.5

            # Run local optimization
            x_opt, f_opt = cma_es(start_x, start_f, per_restart_budget, sigma0, rng)

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 3: Fine-tuning around the best solution
        if time.time() < deadline:
            # Local search with very small steps
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            n_fine = min(50, max(0, int((deadline - time.time()) * 100)))

            for i in range(n_fine):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7
                    sigma = max(sigma, 0.0001)

            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
