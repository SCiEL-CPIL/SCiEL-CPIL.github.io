"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style Schwefel f20 on [-5, 5]^5.
    
    Strategy:
      - Start with a diverse set of initial points.
      - Use a (1+lambda)-ES with adaptive step size and restarts.
      - Final polish pass with scipy L-BFGS-B on the best point found.
      - Deadline guard to ensure we return within 330s wall-clock.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin

        rng = np.random.default_rng(42)

        # Initial points: origin, random, and structured points
        init_points = [np.zeros(dim)]
        for _ in range(6):
            init_points.append(rng.uniform(-5, 5, dim))
        init_points.append(np.full(dim, 4.0))
        init_points.append(np.full(dim, -4.0))
        # Add some mid-range structured points
        init_points.append(np.array([0, 0, 0, 0, 0]) + 1.5 * rng.standard_normal(dim))
        init_points.append(np.array([0, 0, 0, 0, 0]) - 1.5 * rng.standard_normal(dim))

        # Evaluate initial points
        best_x = None
        best_f = np.inf
        for x0 in init_points:
            if time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            budget -= 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            budget -= 1

        # (1+lambda)-ES with adaptive step size
        lam = 15
        sigma = 1.5
        stagnation_count = 0
        restart_threshold = 60
        sigma_min = 0.005
        sigma_max = 5.0

        # Track history of best values for restart decision
        best_history = [best_f]

        while budget > 0 and time.time() < deadline:
            # Generate offspring
            offspring = []
            for _ in range(lam):
                if time.time() > deadline or budget <= 0:
                    break
                # Occasionally restart from a random point to escape local minima
                if rng.random() < 0.1:
                    parent = rng.uniform(-5, 5, dim)
                else:
                    parent = best_x
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                offspring.append(child)

            if not offspring:
                break

            # Evaluate offspring
            offspring_f = []
            for child in offspring:
                if time.time() > deadline or budget <= 0:
                    break
                fc = func(child)
                budget -= 1
                offspring_f.append(fc)

            if not offspring_f:
                break

            # Find best offspring
            best_child_idx = np.argmin(offspring_f)
            best_child = offspring[best_child_idx]
            best_child_f = offspring_f[best_child_idx]

            if best_child_f < best_f:
                best_x = best_child.copy()
                best_f = best_child_f
                stagnation_count = 0
                best_history.append(best_f)
            else:
                stagnation_count += 1
                best_history.append(best_f)

            # Adapt sigma based on success rate
            improved = sum(1 for fc in offspring_f if fc < best_f + 1e-10)
            success_rate = improved / max(1, len(offspring_f))

            if success_rate > 0.25:
                sigma *= 1.15
            elif success_rate < 0.05:
                sigma *= 0.85

            sigma = np.clip(sigma, sigma_min, sigma_max)

            # If stagnated for too long, try a more aggressive restart
            if stagnation_count > restart_threshold:
                # Reduce sigma significantly and do a random restart
                sigma = max(sigma * 0.3, sigma_min)
                # Try a few random restarts
                for _ in range(3):
                    if time.time() > deadline or budget <= 0:
                        break
                    x_try = rng.uniform(-5, 5, dim)
                    f_try = func(x_try)
                    budget -= 1
                    if f_try < best_f:
                        best_x = x_try.copy()
                        best_f = f_try
                        stagnation_count = 0
                        break
                stagnation_count = 0

        # Final polish pass with scipy L-BFGS-B
        if time.time() < deadline - 10 and best_x is not None:
            try:
                # Count how many budget calls we've used
                calls_used = self.budget - budget
                remaining_budget = budget

                if remaining_budget > 20:
                    # Use L-BFGS-B for final polish
                    # Limit the number of function evaluations
                    maxit = min(100, remaining_budget // 2)

                    def wrapped_func(x):
                        nonlocal budget
                        if budget <= 0 or time.time() > deadline:
                            return best_f  # Return current best if out of budget
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        budget -= 1
                        return fx

                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': maxit, 'ftol': 1e-12, 'gtol': 1e-8}
                    )

                    if result.fun < best_f:
                        best_x = np.clip(result.x, -5, 5)
                        best_f = result.fun

            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
