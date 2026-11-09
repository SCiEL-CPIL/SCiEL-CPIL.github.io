"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Halton-seed multi-start local search for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def elapsed():
            return time.time() - start_time

        def time_ok():
            return elapsed() < deadline - 5

        def safe_eval(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or not time_ok():
                return float('inf')
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx if np.isfinite(fx) else float('inf')

        # --- Phase 1: Deterministic Halton sequence for early global coverage ---
        def halton(index, base):
            """Generate a single Halton sequence value."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            """Generate a Halton sequence point in [0,1)^dim."""
            point = np.zeros(dim)
            for d in range(dim):
                # Use different primes for each dimension
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                point[d] = halton(n, primes[d % len(primes)])
            return point

        # Generate Halton points for initial exploration
        n_halton = min(40, budget // 10)
        halton_points = []
        for i in range(1, n_halton + 1):
            pt = halton_point(i, dim)
            # Map to [-5, 5]^dim
            x = pt * 10.0 - 5.0
            fx = safe_eval(x)
            halton_points.append((fx, x.copy()))

        if best_x is None:
            best_x = np.zeros(dim)
            safe_eval(best_x)

        # Sort Halton points by fitness
        halton_points.sort(key=lambda t: t[0])

        # --- Phase 2: Multi-start Nelder-Mead refinement from best Halton points ---
        from scipy.optimize import minimize

        # Select top diverse points for refinement
        # We want points that are not too close to each other
        selected_starts = []
        min_dist = 1.0  # Minimum distance between start points
        for fx, x in halton_points:
            if not time_ok() or calls_made >= budget - 5:
                break
            # Check distance to already selected starts
            too_close = False
            for _, sx in selected_starts:
                if np.linalg.norm(x - sx) < min_dist:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append((fx, x))
            if len(selected_starts) >= 8:
                break

        # If we didn't select enough, add more from the list
        if len(selected_starts) < 5:
            for fx, x in halton_points:
                if len(selected_starts) >= 5:
                    break
                if (fx, x) not in selected_starts:
                    selected_starts.append((fx, x))

        # Run Nelder-Mead from each selected start
        for fx0, x0 in selected_starts:
            if not time_ok() or calls_made >= budget - 3:
                break

            x_start = x0.copy()
            try:
                res = minimize(
                    safe_eval,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 80,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # --- Phase 3: If budget remains, do a few more random restarts with NM ---
        rng = np.random.default_rng(42)
        remaining_budget = budget - calls_made
        if remaining_budget > 20 and time_ok():
            n_extra = min(5, remaining_budget // 15)
            for _ in range(n_extra):
                if not time_ok() or calls_made >= budget - 3:
                    break
                x_rand = rng.uniform(-5, 5, dim)
                try:
                    res = minimize(
                        safe_eval,
                        x_rand,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # --- Phase 4: Final local polish with very tight tolerances ---
        if calls_made < budget - 2 and time_ok() and best_x is not None:
            try:
                res = minimize(
                    safe_eval,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 30,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
