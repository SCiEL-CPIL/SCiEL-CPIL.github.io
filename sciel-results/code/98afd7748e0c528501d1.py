import numpy as np
import time

class Optimizer:
    """
    Optimizer using a deterministic lattice search followed by adaptive local refinement.
    1. A 3^5 grid search over [-5, 5]^5 (243 evaluations) ensures global coverage of the
       multimodal Weierstrass landscape.
    2. A variable-step (1+1)-ES refines the best point found, using the remaining budget
       (up to 257 evaluations) to converge locally.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: 300s limit, kill at 290s to allow for cleanup/return
        deadline = start_time + 290.0

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                f_val = float(func(x))
            except:
                f_val = np.inf
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Lattice Search
        # 3 points per dimension: -5, 0, 5
        # This gives 3^5 = 243 evaluations
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            x = np.array([i * 5 - 5, j * 5 - 5, k * 5 - 5, l * 5 - 5, m * 5 - 5])
                            grid_points.append(x)
        
        for x in grid_points:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)
        
        # Phase 2: Local Refinement with (1+1)-ES
        # Start from the best point found in the lattice search.
        if best_f == np.inf:
            # Fallback if lattice search failed (should not happen)
            best_x = np.zeros(dim)
            evaluate(best_x)

        current_x = best_x.copy()
        current_f = best_f
        sigma = 1.0
        
        # (1+1)-ES Parameters
        # We use the remaining budget.
        # Simple step size adaptation:
        # If step succeeds, increase sigma slightly.
        # If step fails, decrease sigma slightly.
        # This is a robust heuristic for short runs.
        
        while calls < budget and time.time() < deadline:
            # Generate candidate
            z = np.random.randn(dim)
            y = current_x + sigma * z
            
            # Boundary Handling: Clip to [-5, 5]
            y = np.clip(y, -5, 5)
            
            f_y = evaluate(y)
            
            if f_y < current_f:
                # Success
                current_x = y
                current_f = f_y
                best_x = y
                best_f = f_y
                # Increase sigma
                sigma *= 1.1
            else:
                # Failure
                # Decrease sigma
                sigma *= 0.85
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Optional: If sigma is very small, we might be converged.
            # But we keep going to use the budget, as (1+1)-ES is cheap.
            
        return best_x, best_f
