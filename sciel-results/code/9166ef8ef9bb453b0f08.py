"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Start Nelder-Mead with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def count():
            nonlocal total_calls
            total_calls += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            count()
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse random sampling
        n_coarse = min(40, max(15, budget // 15))
        candidates = []
        for i in range(n_coarse):
            if total_calls >= budget - 100:
                break
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            candidates.append((fx, x.copy()))

        if not candidates:
            x = np.zeros(dim)
            evaluate(x)
            candidates = [(best_f, best_x.copy())]

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead from top starting points
        def nelder_mead(start_x, max_calls):
            nonlocal total_calls, best_x, best_f
            dim = self.dim

            # Initialize simplex
            simplex = [start_x.copy()]
            base_step = 0.5
            for i in range(dim):
                v = start_x.copy()
                v[i] += base_step
                v = np.clip(v, -5.0, 5.0)
                simplex.append(v)

            # Evaluate initial simplex
            simplex_f = []
            for v in simplex:
                if total_calls >= budget:
                    return
                f = evaluate(v)
                simplex_f.append(f)

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            while total_calls < budget and (budget - total_calls) > 2 * dim:
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            simplex_f[i] = evaluate(simplex[i])

                        if total_calls >= budget:
                            return

        # Allocate budget among restarts
        remaining = budget - total_calls
        n_restarts = min(6, max(2, remaining // 60))
        calls_per_restart = remaining // n_restarts

        for r in range(n_restarts):
            if total_calls >= budget - 10:
                break
            if r < len(candidates):
                start_x = candidates[r][1].copy()
            else:
                start_x = rng.uniform(-5, 5, dim)
            nelder_mead(start_x, calls_per_restart)

        return best_x, best_f
# EVOLVE-BLOCK-END
