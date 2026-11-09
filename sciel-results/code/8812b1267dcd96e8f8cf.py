"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-phase optimizer: diverse sampling + adaptive (1+1)-ES + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        calls_made = 0

        def elapsed_ok():
            return time.time() < deadline

        def count_call():
            nonlocal calls_made
            calls_made += 1

        # Phase 1: Diverse initial sampling to find a good basin
        # Use a mix of structured and random points
        n_init = min(30, budget // 10)
        best_x = None
        best_f = float('inf')

        # Center point
        x0 = np.zeros(dim)
        fx0 = func(x0)
        count_call()
        best_x = x0.copy()
        best_f = fx0

        # Structured grid-like points at key locations
        init_points = []
        # Origin neighborhood
        for scale in [0.5, 1.0, 2.0, 3.0, 4.0]:
            for d in range(dim):
                for sign in [-1, 1]:
                    pt = np.zeros(dim)
                    pt[d] = sign * scale
                    init_points.append(pt)
        # Random points
        for _ in range(max(0, n_init - len(init_points))):
            init_points.append(rng.uniform(-4, 4, dim))

        for pt in init_points:
            if not elapsed_ok():
                break
            fx = func(pt)
            count_call()
            if fx < best_f:
                best_f = fx
                best_x = pt.copy()

        # Phase 2: Adaptive (1+1)-ES with 1/5 success rule
        x = best_x.copy()
        fx = best_f
        sigma = 1.5  # initial step size
        success_count = 0
        total_tries = 0

        es_budget = budget - calls_made - 20  # reserve 20 for polish
        es_budget = max(es_budget, 50)

        while calls_made < budget - 20 and elapsed_ok():
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            count_call()
            total_tries += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1
            else:
                pass

            # 1/5 success rule for sigma adaptation
            if total_tries >= 5:
                success_rate = success_count / total_tries
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.8
                else:
                    sigma *= 0.95
                success_count = 0
                total_tries = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.005, 4.0)

            # Stagnation detection: if no improvement in last 30 tries, restart locally
            if total_tries == 0 and success_count == 0 and sigma < 0.05:
                # Try a larger jump to escape local optimum
                y = x + 2.0 * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                count_call()
                if fy < fx:
                    x, fx = y, fy
                    sigma = 1.0
                else:
                    sigma = max(sigma, 0.5)

        # Phase 3: Nelder-Mead polish
        polish_budget = budget - calls_made
        if polish_budget > 5 and elapsed_ok():
            # Use Nelder-Mead on the best point found
            try:
                def objective(x_vec):
                    nonlocal calls_made
                    if not elapsed_ok():
                        return best_f  # return current best if out of time
                    result = func(x_vec)
                    count_call()
                    return result

                # Create initial simplex around best point
                x_best = best_x.copy()
                simplex_size = 0.1
                init_simplex = [x_best.copy()]
                for d in range(dim):
                    pt_plus = x_best.copy()
                    pt_plus[d] += simplex_size
                    init_simplex.append(pt_plus)
                    pt_minus = x_best.copy()
                    pt_minus[d] -= simplex_size
                    init_simplex.append(pt_minus)

                # Limit simplex size to stay within budget
                max_simplex = min(len(init_simplex), polish_budget - 1)
                init_simplex = init_simplex[:max_simplex]

                res = minimize(
                    objective,
                    x_best,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(1, polish_budget - len(init_simplex) - 5),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )

                if res.fun < best_f:
                    best_x = res.x.copy()
                    best_f = res.fun

            except Exception:
                # If Nelder-Mead fails, do a simple local refinement
                sigma_polish = 0.05
                for _ in range(min(10, budget - calls_made)):
                    if not elapsed_ok():
                        break
                    y = x + sigma_polish * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    count_call()
                    if fy < fx:
                        x, fx = y, fy
                        best_x = x.copy()
                        best_f = fx
                        sigma_polish *= 0.8
                    else:
                        sigma_polish *= 0.7

        # Update best if ES found something better
        if fx < best_f:
            best_x = x.copy()
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
