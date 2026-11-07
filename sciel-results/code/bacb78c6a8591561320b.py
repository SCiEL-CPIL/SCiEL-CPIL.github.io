"""Optimized BBOB f11 (Discus) solver."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(0)

        # Wrap func to count calls and clip
        calls = 0

        def counted_func(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return func(np.clip(x, -5, 5))

        # Initial point
        x0 = rng.uniform(-5, 5, d)
        best_x = x0.copy()
        best_fx = counted_func(x0)

        # CMA-ES parameters
        # lambda = 4*d + floor(3*ln(d)) for d>=4
        lam = 4 * d + int(3 * np.log(d))
        sigma = 2.0  # initial step size
        c1 = (2 + 2 / (d + 1.7)) / ((d + 1) ** 2 + 2)
        cmu = min(1 - c1, (2 + 2 / (d + 1.7)) / ((d + 1) ** 2 + 2 + 2 * c1))
        cs = (d + 1.7) / (d + 9 + 2 * d / (d + 1.7))
        csigma = cs * (1 + c1 - cmu + 2 / (d - 1))
        damps = d + 2 + 2 / (d - 1)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # CMA-ES state
        mean = x0.copy()
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)

        # Run CMA-ES
        # We'll run until we use up most of the budget or hit time limit
        # Reserve some budget for polish
        cma_budget = int(budget * 0.85)

        while calls < cma_budget and time_ok():
            # Generate lambda points
            B = np.linalg.cholesky(C)
            points = []
            for i in range(lam):
                z = rng.standard_normal(d)
                y = mean + sigma * B @ z
                points.append(y)

            # Evaluate
            fitnesses = []
            for p in points:
                if not time_ok():
                    break
                f = counted_func(p)
                fitnesses.append(f)

            if len(fitnesses) < lam:
                break

            # Sort by fitness
            order = np.argsort(fitnesses)
            points = [points[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]

            # Update best
            if fitnesses[0] < best_fx:
                best_fx = fitnesses[0]
                best_x = points[0].copy()

            # Update mean
            mu = lam // 2
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 1) - np.log(i + 1)
            w = w / w.sum()
            mueff = 1 / np.sum(w ** 2)

            mean_old = mean.copy()
            for i in range(mu):
                mean += w[i] * points[i]

            # Update sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (B @ (mean - mean_old) / sigma)
            sigma *= np.exp(csigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C
            hc = (1 - np.sqrt(1 - (1 - cs) ** 2)) * (np.dot(ps, ps) / d - 1) / (1 - cs)
            pc = (1 - c1) * pc + hc * np.sqrt(c1 * (2 - c1) * mueff) * (mean - mean_old) / sigma

            delta_h = (1 + hc) * (2 - c1) - c1
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (delta_h - 1) * C)
            for i in range(mu):
                yi = B @ (points[i] - mean_old) / sigma
                C += cmu * w[i] * np.outer(yi, yi)

            # Check convergence
            if sigma < 1e-12:
                break

            # Check if we're stuck (no improvement for many iterations)
            if len(fitnesses) > 0 and fitnesses[0] >= best_fx - 1e-10:
                pass  # continue, CMA-ES will adapt

        # Phase 2: Nelder-Mead polish
        if time_ok() and calls < budget - 5:
            try:
                from scipy.optimize import minimize
                polish_budget = budget - calls
                if polish_budget > 5:
                    res = minimize(
                        lambda v: counted_func(np.clip(v, -5, 5)),
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
