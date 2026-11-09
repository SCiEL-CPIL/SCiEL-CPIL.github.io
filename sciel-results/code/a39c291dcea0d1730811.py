"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with deadline guard for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin for 300s limit

        # Track total function calls
        call_count = [0]
        max_calls = budget

        def counted_func(x):
            if call_count[0] >= max_calls:
                return float('inf')
            call_count[0] += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        # Phase 1: Global exploration - sample several random points
        n_global = 8
        best_x = None
        best_f = float('inf')

        for _ in range(n_global):
            if time.time() > deadline or call_count[0] >= max_calls:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            x = np.clip(x, -5.0, 5.0)
            fx = counted_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement from best points using L-BFGS-B
        # Try refining from the best few starting points
        # We'll collect candidate starting points during global phase
        # For simplicity, just refine from the single best point found

        if best_x is not None and time.time() < deadline and call_count[0] < max_calls:
            # Run L-BFGS-B from the best point
            try:
                result = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': max(1, max_calls - call_count[0])}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        # Phase 3: If we have remaining budget, try a few more local refinements
        # from slightly perturbed versions of the best point
        remaining = max_calls - call_count[0]
        if remaining > 20 and time.time() < deadline:
            n_refine = min(3, remaining // 20)
            for i in range(n_refine):
                if time.time() > deadline or call_count[0] >= max_calls:
                    break
                # Perturb the best point
                perturb_scale = 1.0 / (i + 1)
                x_perturbed = best_x + perturb_scale * rng.standard_normal(dim)
                x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                try:
                    result = minimize(
                        counted_func,
                        x_perturbed,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 50, 'maxfun': max(1, max_calls - call_count[0])}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
