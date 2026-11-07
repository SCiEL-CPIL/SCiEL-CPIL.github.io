"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with adaptive population for multi-modal BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf
        bounds = [(-5.0, 5.0)] * self.dim

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float).copy()
            return fx

        def wrapper(x):
            """Wrap func for scipy.optimize.minimize."""
            if calls_made >= self.budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return evaluate(x)

        # Phase 1: Global exploration with multiple random starts
        n_global = min(30, self.budget // 15)
        for i in range(n_global):
            if calls_made >= self.budget or time_left() < 10:
                break
            x0 = rng.uniform(-5, 5, size=self.dim)
            evaluate(x0)

        # Phase 2: Local refinement from best points using L-BFGS-B
        # Collect top candidates for local search
        candidates = []
        if best_x is not None:
            candidates.append(best_x.copy())

        # Add diverse points near the best
        for _ in range(min(10, max(1, self.budget // 30))):
            if best_x is not None and time_left() > 15:
                scale = rng.uniform(0.5, 3.0)
                x_new = np.clip(best_x + scale * rng.standard_normal(self.dim), -5, 5)
                fx = evaluate(x_new)
                candidates.append(x_new.copy())

        # Sort candidates by fitness (re-evaluate if needed)
        cand_fits = []
        for c in candidates:
            if calls_made >= self.budget:
                break
            cf = evaluate(c)
            cand_fits.append((cf, c))
        cand_fits.sort(key=lambda t: t[0])

        # Local optimization from top candidates
        for init_f, init_x in cand_fits[:min(8, len(cand_fits))]:
            if calls_made >= self.budget or time_left() < 5:
                break

            try:
                result = minimize(
                    wrapper,
                    init_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 50}
                )
                if result.success or result.fun < best_f:
                    evaluate(result.x)
            except Exception:
                pass

        # Phase 3: Additional random restarts if budget remains
        while calls_made < self.budget and time_left() > 5:
            if best_x is None:
                x0 = rng.uniform(-5, 5, size=self.dim)
            else:
                scale = rng.uniform(1.0, 5.0)
                x0 = np.clip(best_x + scale * rng.standard_normal(self.dim), -5, 5)
            evaluate(x0)

            # Quick local refinement
            if calls_made < self.budget - 10 and time_left() > 5:
                try:
                    result = minimize(
                        wrapper,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 20, 'maxfun': 20}
                    )
                    if result.fun < best_f:
                        evaluate(result.x)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
