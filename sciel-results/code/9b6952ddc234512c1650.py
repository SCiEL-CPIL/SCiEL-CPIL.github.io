"""Multi-objective optimizer on WFG7: Tchebycheff decomposition with Nelder-Mead local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective search using Tchebycheff decomposition with Nelder-Mead local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s limit

        # Use a deterministic RNG for consistency, or derive from seed if available.
        # The task implies the evaluator sets np.random.seed, but we use our own for internal logic.
        rng = np.random.default_rng(42)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f_arr = np.array(f, dtype=np.float64)
            used += 1
            all_X.append(x.copy())
            all_F.append(f_arr)
            return f_arr

        def is_deadline():
            return time.time() > deadline

        # Phase 1: Initialization
        # We need a good initial set to estimate the ideal point and provide starting points.
        # Use a Latin Hypercube or random sampling.
        n_init = min(20, budget)
        X_init = rng.uniform(lo, hi, size=(n_init, dim))
        for x in X_init:
            evaluate(x)
        
        if not all_F:
            return np.array([])

        F_init = np.array(all_F)
        # Estimate ideal point (lower bound of objectives)
        ideal = np.min(F_init, axis=0)
        
        # Generate Tchebycheff weight vectors
        # For 3 objectives, we can use a grid on the simplex.
        weights = []
        H = 4 # Resolution
        for i in range(H + 1):
            for j in range(H - i + 1):
                k = H - i - j
                w = np.array([i, j, k], dtype=np.float64) / H
                # Avoid zero weights to prevent bias towards axes
                if w[0] > 0.05 and w[1] > 0.05 and w[2] > 0.05:
                    weights.append(w)
        
        # If we don't have enough, add random ones
        while len(weights) < 25:
            w = rng.dirichlet(np.ones(n_obj))
            if np.all(w > 0.05):
                weights.append(w)
                
        weights = np.array(weights)
        
        # Sort weights to prioritize balanced ones (center of the front)
        # Distance from the center (1/3, 1/3, 1/3)
        center = np.ones(n_obj) / n_obj
        dists = np.linalg.norm(weights - center, axis=1)
        sorted_indices = np.argsort(dists)
        weights = weights[sorted_indices]

        # Phase 2: Local Search with Nelder-Mead
        # We have budget - n_init evaluations left.
        # Allocate to weights.
        n_weights = len(weights)
        evals_per_weight = max(3, (budget - used) // n_weights)
        
        # We will iterate through weights.
        # For each weight, we pick the best current solution for that weight as the start.
        # Then we run Nelder-Mead.
        
        # To be safe with the deadline, we check time.
        for w in weights:
            if used >= budget or is_deadline():
                break
            
            # Find best starting point for this weight
            if all_F:
                F_arr = np.array(all_F)
                # Tchebycheff distance: max_i w_i * (f_i - ideal_i)
                # Note: ideal is fixed from init for stability, or updated?
                # Updating ideal can cause instability. Let's use the current min.
                current_ideal = np.min(F_arr, axis=0)
                tcheb_vals = np.max(w * (F_arr - current_ideal), axis=1)
                best_idx = np.argmin(tcheb_vals)
                start_x = all_X[best_idx].copy()
            else:
                start_x = mid.copy()

            # Define objective for Nelder-Mead
            def tcheb_obj(x, w=w, current_ideal=current_ideal):
                # We need to track evaluations here to stop if budget exceeded
                # But Nelder-Mead manages its own internal loop.
                # We must be careful not to exceed the global budget.
                # We will pass a flag or check inside.
                # A safer way: run Nelder-Mead with a limited number of iterations.
                # scipy minimize allows maxiter.
                # However, each iteration calls the function.
                # We can't easily stop scipy from inside without a callback that raises an exception.
                # Let's use a callback to check budget.
                
                nonlocal used
                if used >= budget:
                    # Return a large value to signal failure? Or raise?
                    # Raising might mess up scipy.
                    # Let's just return a large value.
                    return 1e9
                
                x = np.clip(x, lo, hi)
                f = problem(x)
                f_arr = np.array(f, dtype=np.float64)
                used += 1
                all_X.append(x.copy())
                all_F.append(f_arr)
                
                # Calculate Tchebycheff scalarization
                val = np.max(w * (f_arr - current_ideal))
                return val

            # Run Nelder-Mead
            # x0 is start_x
            # maxiter limits the number of function calls roughly.
            # For 12D, Nelder-Mead might need ~20-50 calls to converge.
            # We allocated evals_per_weight.
            # Let's set maxiter to a safe number.
            try:
                res = minimize(tcheb_obj, start_x, method='Nelder-Mead', 
                               options={'maxiter': evals_per_weight, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass

        # Phase 3: Final Polish (Optional, if time/budget remains)
        # The Nelder-Mead should have covered the search space reasonably well.
        
        if len(all_F) > 0:
            return np.array(all_X)
        return np.array([])
# EVOLVE-BLOCK-END
