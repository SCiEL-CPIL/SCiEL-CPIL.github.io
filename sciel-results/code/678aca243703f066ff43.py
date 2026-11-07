"""Hybrid (1+1)-ES with adaptive step size and Nelder-Mead polishing for BBOB f11."""
import numpy as np
import time
import scipy


class Optimizer:
    """Hybrid (1+1)-ES + Nelder-Mead for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Latin hypercube sampling for global exploration
        n_lhc = max(20, budget // 10)  # 50 points for 500 budget
        lhc_points = self._latin_hypercube(n_lhc, dim, rng)
        for x in lhc_points:
            fx = try_eval(x)
            if fx is None:
                break

        if best_x is None:
            x = np.zeros(dim)
            try_eval(x)

        # Sort evaluations to find best starting point for local search
        # We'll track all evaluated points
        # Phase 2: (1+1)-ES from the best point found
        if call_count < budget and best_x is not None:
            self._run_1plus1_es(func, best_x.copy(), budget, dim, call_count, try_eval)

        # Phase 3: Nelder-Mead polishing from the best point
        if call_count < budget and best_x is not None and time.time() < deadline:
            self._run_nelder_mead(func, best_x.copy(), budget, dim, call_count, try_eval)

        return best_x, best_f

    def _latin_hypercube(self, n, dim, rng):
        """Generate n points in [-5, 5]^dim using Latin hypercube sampling."""
        points = []
        for i in range(n):
            x = np.zeros(dim)
            for d in range(dim):
                # For each dimension, sample from a different stratum
                stratum = (i + d) % n
                x[d] = -5.0 + 10.0 * (stratum + rng.random()) / n
            points.append(x)
        return points

    def _run_1plus1_es(self, func, x0, budget, dim, call_count, try_eval):
        """Run (1+1)-ES with adaptive step size."""
        nonlocal best_x, best_f, call_count

        x = x0.copy()
        step_size = 1.0  # initial step size
        success_count = 0
        total_steps = 0
        stagnation_limit = 20  # restart if no progress for this many steps

        # Allocate budget for (1+1)-ES: use up to 60% of remaining budget
        es_budget = int((budget - call_count) * 0.6)
        if es_budget < 20:
            es_budget = max(5, budget - call_count)

        for _ in range(es_budget):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Generate a single mutation
            mutation = rng.standard_normal(dim) * step_size
            x_new = np.clip(x + mutation, -5.0, 5.0)

            fx_new = try_eval(x_new)
            if fx_new is None:
                break

            total_steps += 1
            if fx_new < best_f:
                # Success: accept the new point
                x = x_new.copy()
                success_count += 1
                step_size *= 1.2  # increase step size on success
            else:
                # Failure: keep current point
                step_size *= 0.8  # decrease step size on failure
                # Clamp step size
                step_size = max(1e-6, min(step_size, 5.0))

            # Check for stagnation
            if total_steps > stagnation_limit and success_count / total_steps < 0.1:
                # Reset step size and break out to try Nelder-Mead
                break

        # Update best_x to current x if it's better
        if best_x is not None and x is not None:
            if best_f <= try_eval(x):  # This would use another eval, so just compare
                pass  # best_x is already tracked

    def _run_nelder_mead(self, func, x0, budget, dim, call_count, try_eval):
        """Run Nelder-Mead from the best point."""
        nonlocal best_x, best_f, call_count

        if call_count >= budget:
            return

        # Wrapper that tracks calls
        def func_wrapper(x):
            nonlocal call_count
            if call_count >= budget:
                return best_f if best_f != float('inf') else 1e10
            fx = try_eval(x)
            if fx is None:
                return 1e10
            return fx

        # Use remaining budget for Nelder-Mead
        remaining = budget - call_count
        maxfev = max(10, remaining)

        options = {
            'xatol': 1e-10,
            'fatol': 1e-12,
            'maxiter': 10000,
            'maxfev': maxfev,
            'initial_simplex': None
        }

        try:
            result = scipy.optimize.minimize(
                func_wrapper,
                x0,
                method='Nelder-Mead',
                options=options
            )
            # result.x is the best found by NM, but best_x/best_f are already tracked
        except Exception:
            pass

        return best_x, best_f
