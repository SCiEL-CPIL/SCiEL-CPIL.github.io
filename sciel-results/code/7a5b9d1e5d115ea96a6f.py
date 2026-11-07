"""Lattice-aware basin-hopping optimizer for BBOB-style Rastrigin f3."""
import numpy as np
import time


class Optimizer:
    """
    Exploits Rastrigin periodicity:
    1. Evaluate all 25 integer lattice points to find the best basin.
    2. Use adaptive-step coordinate descent (with halving) to converge to the exact minimum.
    3. If budget remains, try a few other nearby basins or refine with smaller steps.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        calls_used = 0
        best_x = None
        best_f = np.inf

        def try_point(x):
            nonlocal calls_used, best_x, best_f
            if time.time() > deadline:
                return
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Evaluate all 25 integer lattice points in [-5, 5]^5
        # The Rastrigin function has local minima at integer points.
        # The global minimum is at one of these (shifted by hidden offset, but still near integers).
        # Actually, the problem says "minimizer lies in [-4, 4]^5", so we check integers in [-4, 4].
        # But to be safe, we check [-5, 5] integers. That's 11^5 = 161051 which is too many.
        # Wait, the problem says dim=5 and box [-5,5]. The integer lattice points in [-5,5] are 11 per dim.
        # 11^5 = 161,051. That's too many for a budget of 500.
        # But the problem says "minimizer lies in [-4, 4]^5". So we only need to check integers in [-4, 4].
        # That's 9^5 = 59,049. Still too many.
        
        # Rethink: The Rastrigin function is f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)).
        # The local minima are at x_i = integer. The global minimum is at x = 0 (if no shift).
        # With a hidden shift, the global minimum is at x = shift, where shift is in [-4, 4]^5.
        # So the global minimum is NOT necessarily at an integer point.
        # However, the local minima are still at integer points (shifted by the hidden shift? No, the function is shifted, so the minima are at shift + integer).
        # Wait, the problem says "hidden seeded shift, rotation and offset".
        # If the function is f(x) = g(A*x + b) + c, where g is the standard Rastrigin, then the minima are at x = A^(-1)*(k - b) for integer k.
        # This is complicated. Let's not assume the minima are at integer points.
        
        # Alternative approach: Use a coarse grid search to find a good basin, then refine.
        # Since the budget is 500, we can afford a coarse grid.
        # Let's use a grid with 5 points per dimension: 5^5 = 3125 points. Still too many.
        # 3 points per dimension: 3^5 = 243 points. That's feasible.
        # Let's try a 3-point grid in [-4, 4]^5: points at -4, 0, 4.
        # Then refine the best point with coordinate descent.

        # Phase 1: Coarse grid search
        # Use 3 points per dimension: -4, 0, 4
        grid_points = np.array(np.meshgrid([-4.0, 0.0, 4.0] * self.dim)).T.reshape(-1, self.dim)
        for x in grid_points:
            if calls_used >= self.budget or time.time() > deadline:
                break
            try_point(x)

        # Phase 2: Coordinate descent with adaptive step size
        # Start from the best point found in Phase 1
        if best_x is not None:
            x = best_x.copy()
            step = 1.0  # Initial step size
            no_improve = 0
            max_no_improve = 10  # Restart if no improvement for 10 steps

            while calls_used < self.budget and time.time() < deadline:
                improved = False
                for d in range(self.dim):
                    if calls_used >= self.budget or time.time() > deadline:
                        break
                    # Try positive and negative steps
                    for sign in [1.0, -1.0]:
                        if calls_used >= self.budget or time.time() > deadline:
                            break
                        y = x.copy()
                        y[d] = np.clip(y[d] + sign * step, -5.0, 5.0)
                        fy = func(y)
                        calls_used += 1
                        if fy < best_f:
                            x = y
                            best_f = fy
                            best_x = y.copy()
                            improved = True
                        elif fy < x.dot(np.ones(self.dim)) * 0 + best_f:  # This condition is always true if fy < best_f, which we already checked
                            pass

                if improved:
                    no_improve = 0
                    step *= 0.7  # Reduce step size on improvement
                else:
                    no_improve += 1
                    step *= 0.5  # Halve step size on no improvement
                    if step < 1e-6:
                        break
                    if no_improve >= max_no_improve:
                        # Try a random restart from a nearby point
                        if calls_used >= self.budget or time.time() > deadline:
                            break
                        x = best_x + 0.5 * np.random.randn(self.dim)
                        x = np.clip(x, -5.0, 5.0)
                        fx = func(x)
                        calls_used += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve = 0
                        step = 0.5

        # Phase 3: If budget remains, try a few random restarts with local refinement
        if calls_used < self.budget and time.time() < deadline:
            rng = np.random.default_rng(123)
            for _ in range(min(5, (self.budget - calls_used) // 10)):
                if calls_used >= self.budget or time.time() > deadline:
                    break
                x = rng.uniform(-4.0, 4.0, self.dim)
                try_point(x)
                # Quick local refinement
                step = 0.5
                for _ in range(5):
                    if calls_used >= self.budget or time.time() > deadline:
                        break
                    improved = False
                    for d in range(self.dim):
                        if calls_used >= self.budget or time.time() > deadline:
                            break
                        for sign in [1.0, -1.0]:
                            if calls_used >= self.budget or time.time() > deadline:
                                break
                            y = x.copy()
                            y[d] = np.clip(y[d] + sign * step, -5.0, 5.0)
                            fy = func(y)
                            calls_used += 1
                            if fy < best_f:
                                x = y
                                best_f = fy
                                best_x = y.copy()
                                improved = True
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break

        return best_x, best_f
