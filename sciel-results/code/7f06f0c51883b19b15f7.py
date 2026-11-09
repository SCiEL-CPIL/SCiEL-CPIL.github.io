import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 330s limit
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            # Time check
            if time.time() - start_time > time_limit:
                return best_f
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-start L-BFGS-B: gradient-based optimization is far more efficient
        # for Rosenbrock's smooth valley structure than coordinate descent or Nelder-Mead.
        # Each L-BFGS-B run uses ~50-100 function evaluations to converge.
        
        from scipy.optimize import minimize
        
        # Generate diverse starting points: mix of broad random and structured
        # Rosenbrock minimum is at (1,...,1) in standard form; shifted/rotated here.
        # Use a mix of strategies to cover the search space.
        n_starts = 12
        rng = np.random.RandomState(42)
        
        # Phase 1: Broad random sampling (10 points) to find promising basins
        broad_points = rng.uniform(-4.5, 4.5, (10, self.dim))
        
        # Phase 2: Structured points around likely minimum region
        # The minimum is in [-4,4]^5; try points near center and corners
        structured_points = []
        for _ in range(2):
            structured_points.append(rng.uniform(-1, 1, self.dim))
        structured_points = np.array(structured_points)
        
        all_starts = np.vstack([broad_points, structured_points])
        
        # Sort starts by a heuristic: prefer points closer to center (likely valley)
        # but also keep diversity
        center = np.zeros(self.dim)
        dists = np.linalg.norm(all_starts - center, axis=1)
        
        # Interleave: take some from each distance bucket for diversity
        order = np.argsort(dists)
        # Take first 3 (closest), then spread out the rest
        selected_indices = list(order[:3])
        remaining = list(order[3:])
        # Add evenly spaced from remaining for diversity
        if len(remaining) > 0:
            step = max(1, len(remaining) // 9)
            for i in range(0, len(remaining), step):
                if len(selected_indices) >= n_starts:
                    break
                selected_indices.append(remaining[i])
        
        selected_indices = selected_indices[:n_starts]
        starts = all_starts[selected_indices]
        
        # Run L-BFGS-B from each starting point
        for idx in range(len(starts)):
            if calls_made >= self.budget or time.time() - start_time > time_limit:
                break
            
            x0 = starts[idx].copy()
            
            # Check if we've already evaluated this point (unlikely but safe)
            # Just run L-BFGS-B directly
            
            try:
                result = minimize(
                    safe_eval,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={
                        'maxiter': 200,
                        'maxfun': max(5, self.budget - calls_made),
                        'gtol': 1e-6,
                        'ftol': 1e-12,
                    }
                )
            except Exception:
                continue
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
        
        return best_x, best_f
