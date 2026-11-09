"""Deterministic Nelder-Mead simplex optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Solver using Nelder-Mead simplex method with multi-start strategy.
    
    Strategy:
    1. Use a deterministic Nelder-Mead simplex for rapid convergence in ill-conditioned valleys.
    2. Multiple restarts from different initial points to handle the hidden shift/rotation.
    3. Deadline guard to ensure we finish before the time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Nelder-Mead simplex implementation
        def nelder_mead(func_eval, x0, max_iter=100, initial_step=0.5):
            """
            Deterministic Nelder-Mead simplex method.
            Returns (best_x, best_f) found.
            """
            n = len(x0)
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += initial_step

            # Evaluate initial simplex
            f_values = np.array([func_eval(simplex[i]) for i in range(n + 1)])

            # Nelder-Mead parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink

            for iteration in range(max_iter):
                # Sort by function value
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]

                # Check convergence
                if np.ptp(f_values) < 1e-12:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:n], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[n])
                f_ref = func_eval(x_ref)

                if f_ref < f_values[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = func_eval(x_exp)
                    if f_exp < f_ref:
                        simplex[n] = x_exp
                        f_values[n] = f_exp
                    else:
                        simplex[n] = x_ref
                        f_values[n] = f_ref
                elif f_ref < f_values[n - 1]:
                    # Accept reflection
                    simplex[n] = x_ref
                    f_values[n] = f_ref
                else:
                    # Contraction
                    if f_ref < f_values[n]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[n] - centroid)
                    f_con = func_eval(x_con)
                    if f_con < f_values[n]:
                        simplex[n] = x_con
                        f_values[n] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_values[i] = func_eval(simplex[i])

                # Check if we're out of budget
                if calls >= budget:
                    break

            # Find best in simplex
            best_idx = np.argmin(f_values)
            return simplex[best_idx], f_values[best_idx]

        # Multi-start strategy: try several initial points
        rng = np.random.default_rng(42)
        
        # Initial points: center and a few random points
        init_points = [
            np.zeros(dim),  # Center
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            rng.uniform(-3, 3, size=dim),
            rng.uniform(-3, 3, size=dim),
        ]

        # For each initial point, run Nelder-Mead with remaining budget
        for x0 in init_points:
            if calls >= budget or time.time() > deadline:
                break

            # Reserve some budget for this run
            remaining = budget - calls
            if remaining < 10:
                break

            # Run Nelder-Mead
            try:
                x_best_run, f_best_run = nelder_mead(
                    counting_func, 
                    x0.copy(), 
                    max_iter=min(50, remaining // 2),
                    initial_step=1.0
                )
            except Exception:
                break

            # Update global best
            if f_best_run < best_f:
                best_f = f_best_run
                best_x = x_best_run.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
