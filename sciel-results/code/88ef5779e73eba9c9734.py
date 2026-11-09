"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-function-aware optimizer: estimate gradient, jump to optimal vertex, refine."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Track best
        best_x = np.zeros(n)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Sequential coordinate descent to find the optimal vertex
        # For a linear function, the minimum is at a vertex. We can determine
        # each coordinate's optimal bound by comparing f at that bound vs center.
        
        # First, evaluate center to get a baseline
        center = np.zeros(n)
        f_center = safe_call(center)
        if f_center is None:
            return best_x, best_f

        # For each coordinate, determine if minimum is at -5 or +5
        # by evaluating at each bound and comparing
        vertex = np.zeros(n)
        for i in range(n):
            if time.time() > deadline or call_count >= budget:
                break
            
            # Evaluate at -5 for coordinate i (keeping other coords at 0)
            x_neg = np.zeros(n)
            x_neg[i] = -5.0
            f_neg = safe_call(x_neg)
            if f_neg is None:
                break
            
            # Evaluate at +5 for coordinate i (keeping other coords at 0)
            x_pos = np.zeros(n)
            x_pos[i] = 5.0
            f_pos = safe_call(x_pos)
            if f_pos is None:
                break
            
            # Choose the better bound for this coordinate
            if f_neg <= f_pos:
                vertex[i] = -5.0
            else:
                vertex[i] = 5.0

        if time.time() > deadline or call_count >= budget:
            return best_x, best_f

        # Evaluate the candidate vertex
        fv = safe_call(vertex)
        if fv is None:
            return best_x, best_f

        # Phase 2: Local refinement around the best point using adaptive step size
        rng = np.random.default_rng(123)
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        min_sigma = 0.0001
        no_improve_count = 0
        max_no_improve = 20

        while call_count < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(n), -5.0, 5.0)
            fy = safe_call(y)
            if fy is None:
                break
            if fy < fx:
                x, fx = y, fy
                no_improve_count = 0
                # Increase step size slightly on improvement
                sigma = min(sigma * 1.05, 5.0)
            else:
                no_improve_count += 1
                sigma = max(sigma * 0.95, min_sigma)
                if no_improve_count >= max_no_improve:
                    # Try a few more with slightly larger steps before stopping
                    for _ in range(10):
                        if call_count >= budget or time.time() > deadline:
                            break
                        y2 = np.clip(x + 0.05 * rng.standard_normal(n), -5.0, 5.0)
                        fy2 = safe_call(y2)
                        if fy2 is None:
                            break
                        if fy2 < fx:
                            x, fx = y2, fy2
                            no_improve_count = 0
                            sigma = 0.05
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
