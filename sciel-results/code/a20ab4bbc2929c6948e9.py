"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic multi-start with stratified sampling and BFGS local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal total_calls
            if total_calls >= budget:
                return None
            try:
                val = func(x)
                total_calls += 1
                if np.isfinite(val):
                    return float(val)
                return None
            except Exception:
                total_calls += 1
                return None

        def lhd_sample(n_samples, dim, rng):
            """Latin Hypercube Design for stratified sampling."""
            samples = np.zeros((n_samples, dim))
            for d in range(dim):
                # Stratified sampling: divide [0,1] into n_samples strata
                strata = np.linspace(0, 1, n_samples + 1)
                for i in range(n_samples):
                    samples[i, d] = rng.uniform(strata[i], strata[i + 1])
            # Shuffle each dimension independently
            for d in range(dim):
                perm = rng.permutation(n_samples)
                samples[:, d] = samples[perm, d]
            # Map to [-5, 5]
            samples = samples * 10 - 5
            return samples

        def bfgs_local_search(x0, max_evals=50):
            """BFGS local search from x0, using budgeted function calls."""
            nonlocal best_x, best_f, total_calls

            def objective(x):
                nonlocal total_calls
                if total_calls >= budget or time.time() > deadline:
                    return np.inf
                x_clipped = np.clip(x, -5, 5)
                val = safe_call(x_clipped)
                if val is None:
                    return np.inf
                return val

            # Use BFGS with finite difference gradient
            try:
                res = minimize(objective, x0, method='BFGS',
                              options={'maxiter': max_evals, 'gtol': 1e-4})
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

            # Check if we found something better
            if best_x is not None:
                return best_x.copy(), best_f
            return x0, np.inf

        # Generate stratified sample points
        rng = np.random.default_rng(12345)

        # Start with center and a few deterministic points
        initial_points = [np.zeros(dim)]

        # Add stratified Latin Hypercube samples
        n_lhs = 20
        lhs_points = lhd_sample(n_lhs, dim, rng)
        initial_points.extend([p for p in lhs_points])

        # Add some random points for diversity
        n_random = 10
        random_points = rng.uniform(-4, 4, size=(n_random, dim))
        initial_points.extend([p for p in random_points])

        # Sort initial points by distance from center (try center first)
        initial_points.sort(key=lambda p: np.linalg.norm(p))

        # Phase 1: Evaluate all initial points and track best
        for x_init in initial_points:
            if total_calls >= budget or time.time() > deadline:
                break
            x_clipped = np.clip(x_init, -5, 5)
            f_val = safe_call(x_clipped)
            if f_val is not None and f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()

        # Phase 2: BFGS local search from the best few initial points
        if best_x is not None and total_calls < budget and time.time() < deadline:
            # Find top candidates for local refinement
            candidate_points = []
            for x_init in initial_points:
                if total_calls >= budget or time.time() > deadline:
                    break
                x_clipped = np.clip(x_init, -5, 5)
                if len(candidate_points) < 5:
                    candidate_points.append(x_clipped)

            # Run BFGS from each candidate (limit to 5 best)
            for x_cand in candidate_points[:5]:
                if total_calls >= budget or time.time() > deadline:
                    break
                remaining_budget = budget - total_calls
                if remaining_budget < 10:
                    break
                # Allocate remaining budget proportionally
                max_evals_for_this = min(remaining_budget // max(1, len(candidate_points) - len([p for p in candidate_points[:candidate_points.index(x_cand)] if p is not x_cand])), 40)
                bfgs_local_search(x_cand, max_evals=max_evals_for_this)

        # Phase 3: If we have budget left, do a final round of local search from the best point
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining > 20:
                bfgs_local_search(best_x, max_evals=min(remaining, 30))

        # Final fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            if best_f is None:
                best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
