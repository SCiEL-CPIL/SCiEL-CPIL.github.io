"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start scout + Powell polish for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= self.budget:
                return None
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start scouting with candidate points
        # For Rosenbrock, the minimum is often near (1,1,...,1) or shifted
        # Try several strategic starting points
        candidates = []
        
        # Origin
        candidates.append(np.zeros(self.dim))
        
        # Nominal Rosenbrock minimum (1,1,...,1)
        candidates.append(np.ones(self.dim))
        
        # Other strategic points
        candidates.append(np.full(self.dim, 0.5))
        candidates.append(np.full(self.dim, -1.0))
        candidates.append(np.full(self.dim, 2.0))
        candidates.append(np.full(self.dim, -2.0))
        candidates.append(np.full(self.dim, 3.0))
        candidates.append(np.full(self.dim, -3.0))
        
        # Random points within the box
        rng = np.random.default_rng(42)
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, self.dim))

        # Evaluate all candidates
        for cand in candidates:
            if call_count >= self.budget:
                break
            if time.time() > deadline - 120:  # Leave time for local search
                break
            safe_call(cand)

        # Phase 2: Local optimization from the best few starting points
        # Sort candidates by their function values and try local search from top ones
        scored_candidates = []
        for cand in candidates:
            if call_count >= self.budget:
                break
            if time.time() > deadline - 100:
                break
            fx = safe_call(cand)
            if fx is not None:
                scored_candidates.append((fx, cand))

        # Sort by function value
        scored_candidates.sort(key=lambda x: x[0])

        # Try local optimization from the best candidates
        max_local_starts = min(5, len(scored_candidates))
        for i in range(max_local_starts):
            if call_count >= self.budget:
                break
            if time.time() > deadline - 60:
                break
            
            fx0, x0 = scored_candidates[i]
            
            # Powell's method - good for Rosenbrock-like functions
            try:
                result = minimize(
                    lambda x: safe_call(x),
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 50,
                        'maxfev': max(1, self.budget - call_count - 10),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                    }
                )
            except Exception:
                pass

        # Phase 3: Final polish if budget remains
        if best_x is not None and call_count < self.budget and time.time() < deadline - 30:
            try:
                remaining = max(1, self.budget - call_count - 5)
                result = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        # Return best solution found
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = float(func(best_x))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
