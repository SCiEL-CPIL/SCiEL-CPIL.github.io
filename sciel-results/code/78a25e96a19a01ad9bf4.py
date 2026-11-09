"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Multi-strategy optimizer combining CMA-ES-like search with local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        calls = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, step_size=0.5, max_iter=100):
            """Run Nelder-Mead from x0 with given initial step size."""
            nonlocal calls, best_x, best_f, deadline

            n = self.dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += step_size

            simplex_f = np.array([eval_func(simplex[i]) for i in range(n + 1)])

            if calls >= self.budget:
                return

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            it = 0
            while calls < self.budget and it < max_iter:
                if time.time() > deadline:
                    break
                it += 1

                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                    break

                centroid = np.mean(simplex[:n], axis=0)

                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = eval_func(x_r)
                if calls >= self.budget:
                    break

                if f_r < simplex_f[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_func(x_e)
                    if calls >= self.budget:
                        break
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
                    if f_r < simplex_f[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = eval_func(x_c)
                    if calls >= self.budget:
                        break
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = eval_func(simplex[i])
                        if calls >= self.budget:
                            break

        def coordinate_descent(x_start, max_iter=50):
            """Polish x_start using coordinate descent."""
            nonlocal calls, best_x, best_f, deadline
            x = x_start.copy()
            f = eval_func(x)
            step = 0.2
            for _ in range(max_iter):
                if calls >= self.budget or time.time() > deadline:
                    break
                improved = False
                for i in range(self.dim):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        trial = x.copy()
                        trial[i] += sign * step
                        ft = eval_func(trial)
                        if ft < f:
                            x = trial
                            f = ft
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        # Phase 1: Evaluate diverse starting points
        rng = np.random.default_rng(12345)

        init_points = [np.zeros(self.dim)]
        # Add deterministic spread points
        for _ in range(6):
            init_points.append(rng.uniform(-4.5, 4.5, self.dim))
        # Add some structured points
        init_points.append(np.array([3.0] * self.dim))
        init_points.append(np.array([-3.0] * self.dim))
        init_points.append(rng.uniform(-5, 5, self.dim))

        for p in init_points:
            eval_func(p)

        # Phase 2: Run Nelder-Mead from the best point
        if calls < self.budget:
            nelder_mead(best_x.copy(), step_size=1.0, max_iter=100)

        # Phase 3: Coordinate descent polish
        if calls < self.budget:
            coordinate_descent(best_x.copy(), max_iter=30)

        # Phase 4: If budget remains, try additional starts
        if calls < self.budget and time.time() < deadline:
            for _ in range(5):
                if calls >= self.budget or time.time() > deadline:
                    break
                perturbation = rng.normal(0, 1.0, self.dim)
                x_start = np.clip(best_x + perturbation, -self.bound, self.bound)
                eval_func(x_start)
                if calls < self.budget:
                    nelder_mead(x_start, step_size=0.5, max_iter=60)
                if calls < self.budget:
                    coordinate_descent(best_x.copy(), max_iter=20)

        # Phase 5: Final fine-tuning
        if calls < self.budget and time.time() < deadline:
            nelder_mead(best_x.copy(), step_size=0.1, max_iter=60)
            if calls < self.budget:
                coordinate_descent(best_x.copy(), max_iter=30)

        return best_x, best_f
# EVOLVE-BLOCK-END
