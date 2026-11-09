"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and diverse initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit
        dim = self.dim
        total_budget = self.budget

        # Use a seed derived from the function object to vary across instances
        call_seed = hash(id(func)) % (2**31)
        rng = np.random.default_rng(call_seed)

        best_x = None
        best_fx = float('inf')
        calls_made = 0

        # Phase 1: Quick global exploration with multiple diverse starts
        # Phase 2: Focused local refinement from the best point found

        # Define restart strategy: mix of center, moderate, and wide exploration
        num_restarts = 6
        # Allocate budget: more for later refinements
        # First restart gets less (exploration), later ones get more (exploitation)
        budgets = [40, 50, 60, 80, 100, 170]
        # Adjust if sum exceeds budget
        total_alloc = sum(budgets)
        if total_alloc > total_budget:
            scale = total_budget / total_alloc
            budgets = [int(b * scale) for b in budgets]
        # Ensure at least 20 per restart
        budgets = [max(20, b) for b in budgets]

        for restart in range(num_restarts):
            if time.time() > deadline - 2:
                break
            if calls_made >= total_budget:
                break

            budget_here = budgets[restart]
            if calls_made + budget_here > total_budget:
                budget_here = total_budget - calls_made
            if budget_here < 2:
                break

            # Initialization strategy per restart
            if restart == 0:
                # Start at origin (often good for symmetric functions)
                x = np.zeros(dim)
            elif restart == 1:
                # Small random perturbation
                x = rng.uniform(-1, 1, dim)
            elif restart == 2:
                # Moderate exploration
                x = rng.uniform(-2.5, 2.5, dim)
            elif restart == 3:
                # Wider exploration
                x = rng.uniform(-4, 4, dim)
            elif restart == 4:
                # Even wider, biased towards corners
                x = rng.uniform(-4.8, 4.8, dim)
            else:
                # If we have a good best, perturb it; otherwise wide random
                if best_x is not None:
                    x = np.clip(best_x + rng.uniform(-1.5, 1.5, dim), -5, 5)
                else:
                    x = rng.uniform(-4.5, 4.5, dim)

            fx = func(x)
            calls_made += 1

            if fx < best_fx:
                best_fx, best_x = fx, x.copy()

            # Adaptive (1+1)-ES for this restart
            sigma = 2.5
            if restart >= 3:
                sigma = 1.5  # Smaller steps for exploitation phases

            window_size = 10
            window = []

            for _ in range(budget_here - 1):
                if time.time() > deadline:
                    break
                if calls_made >= total_budget:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    window.append(1)
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                else:
                    window.append(0)

                # Adapt sigma
                if len(window) >= window_size:
                    success_rate = sum(window[-window_size:]) / window_size
                    if success_rate > 0.25:
                        sigma *= 1.2
                    elif success_rate < 0.15:
                        sigma *= 0.8
                    window = window[-window_size:]
                else:
                    if len(window) > 5:
                        recent_success = sum(window) / len(window)
                        if recent_success > 0.3:
                            sigma *= 1.1
                        elif recent_success < 0.1:
                            sigma *= 0.85

                sigma = np.clip(sigma, 0.05, 5.0)

        # Final local polish if time and budget allow
        if best_x is not None and calls_made < total_budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.5
            remaining = total_budget - calls_made
            # Do a few more iterations with small sigma
            for _ in range(min(20, remaining)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                sigma *= 0.95

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls_made += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
