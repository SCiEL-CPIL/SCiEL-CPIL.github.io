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
        deadline = start_time + 290.0  # safety margin before 300s limit

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

        # Phase 1: Identify the optimal vertex in 7 calls
        # For a linear function f(x) = g^T x + c, the minimum over [-5,5]^n
        # is at the vertex where x_i = +5 if g_i < 0, else -5.
        # We determine sign(g_i) by comparing f(5*e_i) vs f(0).
        
        # Evaluate center
        center = np.zeros(n)
        f_center = safe_call(center)
        if f_center is None:
            return best_x, best_f

        # For each coordinate, evaluate at +5*e_i to determine gradient sign
        vertex = np.zeros(n)
        for i in range(n):
            if time.time() > deadline or call_count >= budget:
                break
            
            x_pos = np.zeros(n)
            x_pos[i] = 5.0
            f_pos = safe_call(x_pos)
            if f_pos is None:
                break
            
            # If f(5*e_i) < f(0), then g_i < 0, so minimum is at x_i = +5
            # Otherwise g_i >= 0, so minimum is at x_i = -5
            if f_pos < f_center:
                vertex[i] = 5.0
            else:
                vertex[i] = -5.0

        if time.time() > deadline or call_count >= budget:
            return best_x, best_f

        # Evaluate the candidate vertex
        fv = safe_call(vertex)
        if fv is None:
            return best_x, best_f

        # Phase 2: Local refinement with CMA-ES-like adaptive search
        rng = np.random.default_rng(42)
        x = best_x.copy()
        fx = best_f
        sigma = 0.1
        min_sigma = 1e-6
        no_improve_count = 0
        max_no_improve = 50
        pop_size = 10

        # Use a simple population-based refinement
        while call_count < budget and time.time() < deadline:
            # Generate population
            improved = False
            for _ in range(pop_size):
                if call_count >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5.0, 5.0)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    no_improve_count = 0
            if not improved:
                no_improve_count += 1
                sigma *= 0.8
                if sigma < min_sigma or no_improve_count > max_no_improve:
                    break

        # Phase 3: Final high-precision polish
        if call_count < budget and time.time() < deadline:
            # Try very small steps around the best point
            polish_sigma = 1e-4
            for _ in range(min(50, budget - call_count)):
                if time.time() > deadline:
                    break
                for i in range(n):
                    if call_count >= budget or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        if call_count >= budget or time.time() > deadline:
                            break
                        y = x.copy()
                        y[i] += direction * polish_sigma
                        y = np.clip(y, -5.0, 5.0)
                        fy = safe_call(y)
                        if fy is None:
                            break
                        if fy < fx:
                            x, fx = y, fy

        return best_x, best_f
# EVOLVE-BLOCK-END
