"""Coarse-to-fine 1D grid search exploiting Rastrigin separability, then Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine 1D coordinate search + Nelder-Mead polish for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        # Budget tracking
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                raise ValueError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        dim = self.dim

        # Phase 1: Coarse-to-fine 1D grid search along each axis
        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global minimum at each x_i = 0 (in local coords), so we search for the
        # basin by finding the best 1D slice for each coordinate.

        # Initialize with center
        x = np.zeros(dim)
        fx = call_func(x)

        # Coarse 1D scan for each axis
        # 11 points per axis (step=1.0, aligned with Rastrigin period)
        coarse_points = np.linspace(-5, 5, 11)

        for axis in range(dim):
            best_val = fx
            best_pt = x[axis]
            for pt in coarse_points:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val

        # Medium grid: 21 points in a window of width 2 around current best for each axis
        for axis in range(dim):
            center = x[axis]
            window = np.linspace(max(-5, center - 1.0), min(5, center + 1.0), 21)
            best_val = fx
            best_pt = x[axis]
            for pt in window:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val

        # Check time
        if time.time() > deadline:
            return x, fx

        # Fine grid: 31 points in a window of width 0.4
        for axis in range(dim):
            center = x[axis]
            window = np.linspace(max(-5, center - 0.2), min(5, center + 0.2), 31)
            best_val = fx
            best_pt = x[axis]
            for pt in window:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val

        if time.time() > deadline:
            return x, fx

        # Very fine grid: 15 points in a window of width 0.04
        for axis in range(dim):
            center = x[axis]
            window = np.linspace(max(-5, center - 0.02), min(5, center + 0.02), 15)
            best_val = fx
            best_pt = x[axis]
            for pt in window:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val

        if time.time() > deadline:
            return x, fx

        # Phase 2: Nelder-Mead polish
        from scipy.optimize import minimize

        best_x = x.copy()
        best_f = fx

        # Nelder-Mead from current best
        try:
            result = minimize(
                call_func,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass

        # Check time
        if time.time() > deadline:
            return best_x, best_f

        # If we still have budget, do a second Nelder-Mead with even tighter tolerance
        if calls_made < self.budget - 10:
            try:
                result = minimize(
                    call_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        # Check time
        if time.time() > deadline:
            return best_x, best_f

        # Multi-start: try small perturbations around best_x
        if calls_made < self.budget - 50:
            rng = np.random.default_rng(42)
            for attempt in range(3):
                if calls_made >= self.budget - 5:
                    break
                if time.time() > deadline:
                    break
                perturbation = rng.normal(0, 0.01, size=dim)
                start_x = np.clip(best_x + perturbation, -5, 5)
                try:
                    result = minimize(
                        call_func,
                        start_x,
                        method='Nelder-Mead',
                        options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                    )
                    if result.fun < best_f:
                        best_x = result.x
                        best_f = result.fun
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
