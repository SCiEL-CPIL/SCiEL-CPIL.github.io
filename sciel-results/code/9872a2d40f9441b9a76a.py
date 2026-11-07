import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es_1plus1(x0, f0, budget_left, sigma0=1.0):
            """
            1+1 CMA-ES for local refinement.
            """
            nonlocal calls, best_x, best_f
            if budget_left < 10:
                return x0, f0

            x = x0.copy()
            f = f0
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            cs = (n + 2) / (n + 8)
            cmu = 2 / ((n + 4) ** 2 + 1)
            cc = (4 + n / 4) / (n + 4 + 2 / cs)
            damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            for _ in range(budget_left):
                if calls >= budget - 2 or time.time() > deadline - 2:
                    break

                # Sample
                z = rng.standard_normal(n)
                y = sigma * z
                x_trial = x + y

                f_trial = try_eval(x_trial)
                if f_trial is None:
                    break

                if f_trial < f:
                    # Success
                    y = x_trial - x
                    step = y / sigma
                    pc = (1 - cs) * pc + cs * step
                    norm_pc = np.linalg.norm(pc)
                    hs = 1 if (norm_pc / np.sqrt(1 - (1 - cs) ** (2 * (calls - budget_left + 1))) <
                               (1.4 + 1 / n) * chi_n) else 0
                    C = (1 - cmu * hs) * C + cmu * hs * np.outer(step, step)
                    x = x_trial
                    f = f_trial
                    # Update sigma
                    sigma *= np.exp(cs / damps * (norm_pc / chi_n - 1))
                    # Adaptive step size
                    sigma = np.clip(sigma, 1e-8, 10.0)
                else:
                    # Failure - just keep going
                    pass

            return x, f

        # Phase 1: Sobol-like quasi-random global sampling
        # Use a simple Halton sequence for low-discrepancy coverage
        def halton(index, dim):
            """Generate Halton sequence point."""
            point = []
            for j in range(dim):
                prime = [2, 3, 5, 7, 11][j]
                f = 1.0
                result = 0.0
                i = index
                while i > 0:
                    f /= prime
                    result += f * (i % prime)
                    i //= prime
                point.append(result * 10.0 - 5.0)
            return np.array(point)

        # Sample 80 points with Halton sequence
        n_global = 80
        for i in range(1, n_global + 1):
            if calls >= budget - 50 or time.time() > deadline - 30:
                break
            x = halton(i, n)
            try_eval(x)

        # Add a few random points for diversity
        for _ in range(10):
            if calls >= budget - 50 or time.time() > deadline - 30:
                break
            x = rng.uniform(-5, 5, n)
            try_eval(x)

        # Phase 2: Local refinement with 1+1 CMA-ES from best point
        if best_f < float('inf') and calls < budget - 10:
            remaining = budget - calls
            # Use most of remaining budget for CMA-ES
            cma_budget = min(remaining - 5, 300)
            if cma_budget >= 10:
                x_best, f_best = cma_es_1plus1(best_x, best_f, cma_budget)
                if f_best < best_f:
                    best_x = x_best
                    best_f = f_best

        # Phase 3: If we have budget left, try a second CMA-ES from a different starting point
        if calls < budget - 10 and time.time() < deadline - 10:
            # Find second-best distinct point
            if calls < budget - 10:
                # Generate a few more candidate starts
                candidates = []
                for i in range(20, 60):
                    if calls >= budget - 10 or time.time() > deadline - 10:
                        break
                    x = halton(i, n)
                    f = try_eval(x)
                    if f is not None:
                        candidates.append((f, x))

                if candidates:
                    candidates.sort(key=lambda t: t[0])
                    # Pick the best candidate that is far from current best
                    for f_c, x_c in candidates:
                        if np.linalg.norm(x_c - best_x) > 1.0:
                            remaining = budget - calls
                            if remaining >= 10:
                                cma_budget = min(remaining - 5, 100)
                                if cma_budget >= 10:
                                    x_try, f_try = cma_es_1plus1(x_c, f_c, cma_budget)
                                    if f_try < best_f:
                                        best_x = x_try
                                        best_f = f_try
                            break

        return best_x, best_f
