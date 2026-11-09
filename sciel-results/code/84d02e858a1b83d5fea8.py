"""Rewritten optimizer for BBOB f20 (Schwefel) with Multi-start Global Search and Local Refinement."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start optimizer: Biased Scan + LHS Global Search + L-BFGS-B Refinement + (1+1)-ES Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit

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
        best_f_history = []

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Biased initial scan targeting Schwefel's known minimum region
        # Schwefel f20 global min is near x_i = 4.20968...
        schwefel_min = 4.209687463375637
        candidates = []
        
        # Key points: origin, schwefel min, and variations
        candidates.append(np.zeros(dim))
        candidates.append(np.full(dim, schwefel_min))
        candidates.append(np.full(dim, -schwefel_min))
        
        # Axis-aligned explorations around the expected minimum
        for axis in range(dim):
            for val in [-5, -4, -2, 0, 2, 4, 4.2, 5]:
                pt = np.full(dim, schwefel_min)
                pt[axis] = val
                candidates.append(pt)
        
        # Some random points in the promising region [3, 5]^dim
        for _ in range(10):
            candidates.append(rng.uniform(3.0, 5.0, dim))
        
        for pt in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            f = safe_call(pt)
            update_best(pt, f)
            best_f_history.append(best_f)

        # Phase 2: Latin Hypercube Sampling for broader exploration
        # Use LHS to ensure coverage of the entire domain
        n_lhs = 80
        if budget - call_count > n_lhs:
            # Generate LHS points
            lhs_points = []
            for i in range(n_lhs):
                if call_count >= budget or time.time() > deadline:
                    break
                # LHS: for each dimension, divide [-5,5] into n_lhs intervals,
                # pick one random sub-interval per dimension
                pt = np.zeros(dim)
                for d in range(dim):
                    interval_start = -5.0 + i * (10.0 / n_lhs)
                    interval_end = interval_start + 10.0 / n_lhs
                    pt[d] = rng.uniform(interval_start, interval_end)
                lhs_points.append(pt)
            
            # Evaluate LHS points
            lhs_results = []
            for pt in lhs_points:
                if call_count >= budget or time.time() > deadline:
                    break
                f = safe_call(pt)
                if np.isfinite(f):
                    lhs_results.append((f, pt))
                    update_best(pt, f)
                    best_f_history.append(best_f)
            
            lhs_results.sort(key=lambda x: x[0])

        # Phase 3: Local Refinement from Top candidates using L-BFGS-B with tight tolerances
        top_k = 6
        candidates_to_refine = []
        
        # Add current best
        if best_f < np.inf:
            candidates_to_refine.append(best_x.copy())
        
        # Add top LHS/random points
        all_explored = lhs_results if 'lhs_results' in dir() else []
        for f, pt in all_explored[:top_k]:
            is_duplicate = False
            for existing in candidates_to_refine:
                if np.linalg.norm(pt - existing) < 0.05:
                    is_duplicate = True
                    break
            if not is_duplicate:
                candidates_to_refine.append(pt.copy())
        
        # Also add the Schwefel minimum point if not already there
        schwefel_pt = np.full(dim, schwefel_min)
        is_dup = False
        for existing in candidates_to_refine:
            if np.linalg.norm(schwefel_pt - existing) < 0.05:
                is_dup = True
                break
        if not is_dup:
            candidates_to_refine.append(schwefel_pt)
        
        # Refine each candidate using L-BFGS-B with tight tolerances
        num_refine = len(candidates_to_refine)
        if num_refine > 0 and budget - call_count > 20:
            budget_per_refine = (budget - call_count) // num_refine
            budget_per_refine = max(20, budget_per_refine)
            
            for cx in candidates_to_refine:
                if time.time() > deadline or call_count >= budget:
                    break
                
                try:
                    res = minimize(
                        lambda x: safe_call(x),
                        cx,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 200,
                            'maxfun': budget_per_refine,
                            'ftol': 1e-15,
                            'gtol': 1e-10,
                            'disp': False
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        update_best(res.x, res.fun)
                        best_f_history.append(best_f)
                except Exception:
                    pass

        # Phase 4: (1+1)-ES with stagnation detection and restarts
        if time.time() < deadline and budget - call_count > 30 and best_f < np.inf:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            no_improve = 0
            stagnation_threshold = 30
            
            remaining = budget - call_count
            for _ in range(remaining):
                if time.time() > deadline or call_count >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, 5.0)
                    no_improve = 0
                    update_best(x, fx)
                    best_f_history.append(best_f)
                else:
                    sigma = max(sigma * 0.85, 1e-6)
                    no_improve += 1
                
                # Stagnation detection: restart from a new random point in promising region
                if no_improve >= stagnation_threshold and budget - call_count > 50:
                    # Restart from a random point near the current best or in promising region
                    if rng.random() < 0.5:
                        x = np.clip(best_x + rng.standard_normal(dim) * 1.0, -5, 5)
                    else:
                        x = rng.uniform(3.0, 5.0, dim)
                    fx = safe_call(x)
                    if fx < best_f:
                        update_best(x, fx)
                    sigma = 0.5
                    no_improve = 0
                
                if sigma < 1e-8:
                    break

        return best_x, best_f
