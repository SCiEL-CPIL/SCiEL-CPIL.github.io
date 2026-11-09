"""Rewritten optimizer for BBOB f20 (Schwefel) with Multi-start Global Search and Local Refinement."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start optimizer: Structured Scan + Random Global Search + L-BFGS-B Refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270  # Safety margin for 300s limit

        rng = np.random.default_rng(42)
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            if not np.isfinite(fx):
                return np.inf
            call_count += 1
            return fx

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Structured initial scan (uses ~30 calls)
        # Try origin, axis-aligned points
        candidates = [np.zeros(dim)]
        for axis in range(dim):
            for val in [-4, -2, 0, 2, 4]:
                pt = np.zeros(dim)
                pt[axis] = val
                candidates.append(pt)
        
        for pt in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            f = safe_call(pt)
            update_best(pt, f)

        # Phase 2: Random Global Search with diverse starting points
        # Use a mix of random points to explore the landscape
        n_random = 150
        if budget - call_count > n_random:
            random_results = []
            for i in range(n_random):
                if call_count >= budget or time.time() > deadline:
                    break
                # Generate random points in [-5, 5]^dim
                pt = rng.uniform(-5.0, 5.0, dim)
                f = safe_call(pt)
                if np.isfinite(f):
                    random_results.append((f, pt))
            
            # Sort by fitness
            random_results.sort(key=lambda x: x[0])
            
            # Phase 3: Local Refinement from Top K random points
            # We take the top 8 points from random search and the current best
            top_k = 8
            candidates_to_refine = []
            
            # Add current best
            if best_f < np.inf:
                candidates_to_refine.append(best_x)
            
            # Add top random points
            for f, pt in random_results[:top_k]:
                # Check if this point is close to existing candidates to avoid redundancy
                is_duplicate = False
                for existing in candidates_to_refine:
                    if np.linalg.norm(pt - existing) < 0.1:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    candidates_to_refine.append(pt)
            
            # Refine each candidate using L-BFGS-B
            # Budget remaining: budget - call_count
            # Split among candidates
            num_refine = len(candidates_to_refine)
            if num_refine > 0 and budget - call_count > 10:
                budget_per_refine = (budget - call_count) // num_refine
                # Ensure at least 10 calls per refinement
                budget_per_refine = max(10, budget_per_refine)
                
                for cx in candidates_to_refine:
                    if time.time() > deadline or call_count >= budget:
                        break
                    
                    # Use L-BFGS-B for local optimization
                    # It is fast and effective for smooth functions like Schwefel (after finding the basin)
                    try:
                        res = minimize(
                            lambda x: safe_call(x),
                            cx,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 100, 'maxfun': budget_per_refine}
                        )
                        if res.fun < best_f:
                            update_best(res.x, res.fun)
                    except Exception:
                        pass

            # Phase 4: Final Polish with (1+1)-ES if budget remains
            # L-BFGS-B might stop at a local minimum or boundary. A small step-size search can help.
            if time.time() < deadline and budget - call_count > 20 and best_f < np.inf:
                x = best_x.copy()
                fx = best_f
                sigma = 0.2 # Start with small sigma
                remaining = budget - call_count
                for _ in range(remaining):
                    if time.time() > deadline or call_count >= budget:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = safe_call(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                        update_best(x, fx)
                    else:
                        sigma *= 0.9
                    if sigma < 1e-8:
                        break

        return best_x, best_f
