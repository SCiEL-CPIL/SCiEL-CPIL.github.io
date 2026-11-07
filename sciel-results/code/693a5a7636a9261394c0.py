"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer with strict budget enforcement and structured exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_fx = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_fx
            if call_count >= budget:
                return 1e10
            call_count += 1
            if call_count > budget:
                raise ValueError("Budget exceeded")

            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            try:
                fx = float(func(x))
            except Exception:
                return 1e10

            if not np.isfinite(fx):
                return 1e10

            if fx < best_fx:
                best_fx = fx
                best_x = np.array(x, dtype=float).copy()

            return fx

        # Generate diverse initial points using a structured approach
        # This helps in navigating the narrow valley of the Discus function
        # by providing candidates along axes and random points.
        rng = np.random.default_rng(42)

        x0_candidates = []

        # Include the center
        x0_candidates.append(np.zeros(d))

        # Add axis-aligned points (along each dimension at several levels)
        # This helps if the minimum is shifted along an axis
        for dim_idx in range(d):
            for level in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                pt = np.zeros(d)
                pt[dim_idx] = level
                x0_candidates.append(pt)

        # Add random points to cover the space more broadly
        n_random = 10
        for _ in range(n_random):
            x0 = rng.uniform(-5, 5, size=d)
            x0_candidates.append(x0)

        # Define bounds for L-BFGS-B
        bounds = [(-5.0, 5.0)] * d

        # Phase 1: Evaluate all candidates and sort them
        # We do this to prioritize local refinement on the most promising starting points
        # Note: wrapped_func updates best_x/best_fx automatically
        candidate_fits = []
        for x0 in x0_candidates:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break
            fx = wrapped_func(x0)
            candidate_fits.append((fx, x0))

        # Sort by fitness (ascending)
        candidate_fits.sort(key=lambda t: t[0])

        # Phase 2: Local refinement with L-BFGS-B on the best candidates
        # We allocate the remaining budget to refine the top candidates.
        # L-BFGS-B is efficient for smooth functions like Discus.
        # We limit the number of refinements and iterations to stay within budget.

        # Estimate calls per L-BFGS-B run. In 5D, it's usually 20-50 calls per run.
        # We have budget - call_count left.
        remaining_budget = budget - call_count

        # We want to try refining the top few candidates.
        # Let's say we try up to 5 candidates.
        # Each candidate gets a share of the remaining budget.
        # But we also need to stop if we run out of time or budget.

        n_refine = 5
        # Rough estimate: 30 calls per L-BFGS-B run?
        # Let's just run them and let the wrapped_func stop us if we exceed budget.
        # However, minimize might not respect the budget strictly if it doesn't call the function.
        # But L-BFGS-B calls the function frequently.

        for i in range(min(n_refine, len(candidate_fits))):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            f0, x0 = candidate_fits[i]

            # If the starting point is already very good, we might still want to refine
            # to ensure we are at the local minimum.

            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,  # Limit iterations to prevent runaway
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                # minimize returns the best found, but our wrapped_func tracks the global best
                # so best_x and best_fx are already updated if improvement was found.

            except ValueError as e:
                if "Budget exceeded" in str(e):
                    break
                else:
                    raise e
            except Exception:
                # If optimization fails for other reasons, continue to next start
                continue

        # Phase 3: If we still have budget, do a final polish from the current best
        if time.time() < deadline and call_count < budget - 10:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 30,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
            except Exception:
                pass

        if best_x is None:
            # Fallback if no calls were made or all failed
            best_x = np.zeros(d)
            try:
                best_fx = wrapped_func(best_x)
            except:
                best_fx = 0.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
