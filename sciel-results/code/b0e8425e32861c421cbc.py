"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Deterministic multi-start Powell optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        # Count function calls
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            # Check deadline
            if time.time() > deadline:
                raise TimeoutError()
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x_clipped))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Generate stratified starting points using Sobol sequence
        # We'll use a simple quasi-random approach: stratified grid + jitter
        dim = self.dim
        budget = self.budget

        # Generate initial points: stratified across the domain
        # Use a combination of center, corners, and quasi-random points
        n_starts = min(50, budget // 10)  # up to 50 restarts, each using ~10% budget

        # Sobol-like points using golden ratio stratification
        starts = []
        # Include origin and some structured points
        starts.append(np.zeros(dim))
        starts.append(np.full(dim, 0.5))
        starts.append(np.full(dim, -0.5))

        # Add quasi-random points
        golden = 0.6180339887498949
        for i in range(1, 30):
            t = (i * golden) % 1.0
            pt = np.zeros(dim)
            for d in range(dim):
                # Use different golden ratios for different dimensions
                ratio = 0.6180339887498949 + 0.1 * d
                val = ((i * ratio) % 1.0) * 10.0 - 5.0
                pt[d] = val
            starts.append(pt)

        # Add some random points
        rng = np.random.default_rng(42)
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, dim))

        # Sort starts by some heuristic: prefer points closer to center
        # (Discus function often has minimum near center after transformations)
        starts.sort(key=lambda p: np.linalg.norm(p))

        # Budget allocation per restart
        # Powell method typically uses O(dim * iterations) function evaluations
        # Allocate budget dynamically
        remaining_budget = budget
        restarts_done = 0

        for start_idx, x0 in enumerate(starts):
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Check budget
            if call_count >= budget:
                break

            # Determine max iterations for this restart
            # Powell in 5D: each iteration does ~5 line searches, each ~5-10 evals
            # So ~25-50 evals per iteration. Allow ~budget/10 evals per restart.
            evals_allowed = max(20, (budget - call_count) // (len(starts) - restarts_done))
            max_iter = max(10, evals_allowed // 30)  # conservative estimate

            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': max_iter,
                        'maxfev': evals_allowed,
                        'xtol': 1e-10,
                        'ftol': 1e-12,
                    }
                )

                # Update best if this restart found something better
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)

            except TimeoutError:
                break
            except Exception:
                # If minimize fails, continue to next restart
                pass

            restarts_done += 1

            # If we've used most of the budget, stop
            if call_count >= budget - 50:
                break

        # If we have budget left, do a final local refinement from the best point
        if best_x is not None and call_count < budget - 20 and time.time() < deadline - 5:
            remaining = budget - call_count
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining,
                        'xtol': 1e-12,
                        'ftol': 1e-14,
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except (TimeoutError, Exception):
                pass

        if best_x is None:
            # Fallback: should not happen, but just in case
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
