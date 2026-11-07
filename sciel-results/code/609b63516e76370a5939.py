"""Global-to-local CMA-ES optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global random search followed by single-run CMA-ES with adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s wall-clock limit

        dim = self.dim
        total_budget = self.budget

        n_calls = 0
        best_f = float('inf')
        best_x = None

        rng = np.random.default_rng(42)

        def record(x, fx):
            nonlocal n_calls, best_f, best_x
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float, copy=True)

        # Phase 1: Extensive random exploration to identify the best basin
        # Use 50 uniform random evaluations to find a good starting point
        n_initial = 50
        for _ in range(n_initial):
            if n_calls >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            record(x, fx)

        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Phase 2: Single CMA-ES run from the best point found
        # Use remaining budget for focused local optimization
        
        lambda_pop = 10
        mu = lambda_pop // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Start CMA-ES from the best point found in Phase 1
        x = best_x.copy()
        sigma = 1.0  # Initial step size
        
        # CMA-ES state
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        gen = 0
        best_in_run = best_f
        no_improve_count = 0
        stagnation_limit = dim * 10  # Restart if no improvement for this many generations
        
        # Run CMA-ES until budget or time runs out
        while n_calls < total_budget and time.time() < deadline:
            # Check for stagnation
            if no_improve_count > stagnation_limit:
                break
                
            # If a full population no longer fits, use remaining for local search
            if n_calls + lambda_pop > total_budget:
                pop_size = total_budget - n_calls
                if pop_size < 1:
                    break
                for _ in range(pop_size):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = func(y)
                    record(y, fy)
                break

            # Generate population
            ys = []
            fs = []
            for j in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                record(y, fy)
                ys.append(y)
                fs.append(fy)

            ys = np.array(ys)
            fs = np.array(fs)
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Check for improvement in this generation
            if fs[0] < best_in_run:
                best_in_run = fs[0]
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Weighted mean
            m_new = np.sum(weights[:, None] * ys[:mu], axis=0)

            # Update CMA-ES state
            y_sigma = (m_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n) < (1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + hsig * c_c * y_sigma
            delta_hsig = (1 - hsig) * c_c * (1 - c_c)

            v_new = np.zeros((dim, dim))
            for i in range(mu):
                v_new += weights[i] * np.outer(ys[i] - m_new, ys[i] - m_new)
            
            C = (1 + c_1 * delta_hsig + c_mu - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + delta_hsig) \
                + c_mu * v_new / (sigma ** 2)

            # Ensure positive definite
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-8 * np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ys[0] - m_new) / sigma - 1))
            sigma = np.clip(sigma, 0.001, 5.0)

            # Move center
            x = m_new
            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
