"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start: coarse sampling -> fast coordinate sweep -> adaptive ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s/330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
                calls_used += 1
                return val
            except Exception:
                calls_used += 1
                return np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return calls_used < budget and time.time() <= deadline

        def fast_coordinate_sweep(x, fx):
            """
            Fast coordinate descent using a few fixed step sizes.
            This is much faster than binary search for initial convergence.
            """
            nonlocal calls_used
            x = x.copy()
            fx = fx
            # Step sizes to try: small for fine-tuning, large for escaping
            steps = [0.5, 1.0, 2.0, 4.0]
            
            # Iterate through dimensions a few times
            # 2 passes is usually enough for separable functions
            for _ in range(2):
                if not time_ok():
                    break
                for d in range(dim):
                    if not time_ok():
                        break
                    base_val = x[d]
                    current_best_step = 0
                    current_best_f = fx
                    
                    for step in steps:
                        for sign in [1, -1]:
                            new_val = base_val + sign * step
                            if new_val < -5.0 or new_val > 5.0:
                                continue
                            y = x.copy()
                            y[d] = new_val
                            fy = safe_call(y)
                            if fy < current_best_f:
                                current_best_f = fy
                                current_best_step = sign * step
                            
                    if current_best_step != 0:
                        x[d] += current_best_step
                        fx = current_best_f
                        update_best(x, fx)
                    else:
                        # If no improvement found with these steps, we assume local min in this dim
                        # We could try smaller steps, but for speed we skip
                        pass
            return x, fx

        def es_refine(x, fx, max_iterations=None):
            """
            Adaptive (1+1)-ES refinement.
            """
            nonlocal calls_used
            x = x.copy()
            fx = fx
            sigma = 1.0
            
            # Run until budget/time is out or sigma becomes tiny
            # max_iterations is a soft cap if provided, but time/budget is hard
            while time_ok():
                if max_iterations is not None and max_iterations <= 0:
                    break
                if max_iterations is not None:
                    max_iterations -= 1
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    # Increase sigma slightly on success to explore more
                    sigma *= 1.05 
                else:
                    # Decrease sigma on failure to converge
                    sigma *= 0.85
                
                if sigma < 1e-6:
                    break

            return x, fx

        # ---- Phase 1: Coarse random sampling ----
        # 25 points is a good balance. Not too many to waste, not too few to miss regions.
        n_coarse = 25
        coarse_x = []
        coarse_f = []
        for _ in range(n_coarse):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if np.isfinite(fx):
                coarse_x.append(x)
                coarse_f.append(fx)

        if not coarse_x:
            x0 = np.zeros(dim)
            fx0 = safe_call(x0)
            update_best(x0, fx0 if np.isfinite(fx0) else 0.0)
            return best_x, best_f

        # Sort by fitness
        order = np.argsort(coarse_f)
        k = min(5, len(coarse_x))
        top_indices = order[:k]

        # Initialize best with the best coarse point
        best_x = coarse_x[top_indices[0]].copy()
        best_f = coarse_f[top_indices[0]]

        # ---- Phase 2 & 3: Refinement ----
        # Strategy: 
        # 1. Heavy refinement on the best start.
        # 2. Lighter refinement on the next few starts to escape local minima.
        
        remaining = budget - calls_used
        
        # Weights: 50% for best, 20% for 2nd, 15% for 3rd, 10% for 4th, 5% for 5th
        weights = [0.5, 0.2, 0.15, 0.1, 0.05]
        # Normalize if k < 5
        weights = weights[:k]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        for i, idx in enumerate(top_indices):
            if not time_ok():
                break
            
            x = coarse_x[idx].copy()
            fx = coarse_f[idx]
            
            # Allocate budget for this restart
            # We don't strictly track calls per restart, but we limit the ES iterations
            # based on the weight to ensure we finish all top starts if possible.
            # However, the ES loop breaks on time_ok() anyway.
            # To be safe, we can pass a max_iter estimate.
            alloc_calls = int(remaining * weights[i])
            
            # 1. Fast coordinate sweep (cheap, deterministic improvement)
            x, fx = fast_coordinate_sweep(x, fx)
            
            # 2. ES Refinement
            # Estimate max iterations for ES. 
            # If we have 100 calls allocated, and ES uses ~1 call/iter, max_iter=100.
            # But we should leave some buffer.
            max_es_iters = max(10, alloc_calls // 2) 
            x, fx = es_refine(x, fx, max_es_iters)
            
            # Update remaining budget estimate for subsequent loops (approx)
            # We don't strictly need to update 'remaining' as time_ok handles the hard stop,
            # but it helps with the max_es_iters calculation if we were doing strict accounting.
            # Since we rely on time_ok, we just proceed.

        # If we have significant budget left (e.g., if convergence was very fast),
        # do a final global restart or a very tight ES on the best point.
        if time_ok() and (budget - calls_used) > 20:
             # Final polish on the global best
             x = best_x.copy()
             fx = best_f
             # Small sigma for final polish
             # Re-use es_refine but it will stop when budget is out
             # To ensure we use the last few calls effectively, we can force a small sigma
             # by just running the standard es_refine which adapts sigma anyway.
             es_refine(x, fx, max_iterations=(budget - calls_used))

        return best_x, best_f
# EVOLVE-BLOCK-END
