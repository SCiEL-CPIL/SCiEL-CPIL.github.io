"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for BBOB f13 sharp ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 320s margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)
        
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                raise Exception("Budget exhausted")
            calls_made += 1
            return func(x)

        # Phase 1: Global multi-start sampling
        n_init = min(50, budget // 10)
        if n_init < 10:
            n_init = min(10, budget)
        
        candidates = []
        for i in range(n_init):
            if calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            candidates.append((fx, x.copy()))
            if time.time() > deadline - 30:
                break

        if not candidates:
            return np.zeros(dim), float('inf')

        # Sort by fitness and take top 5
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))
        
        # If we barely have budget, just return best
        remaining = budget - calls_made
        if remaining <= 0 or top_k == 0:
            return candidates[0][1], candidates[0][0]

        # Phase 2: Run (1+1)-ES from each top starting point
        # Allocate remaining budget evenly among the top_k starts
        budget_per_start = max(1, remaining // top_k)
        
        global_best_x = candidates[0][1].copy()
        global_best_f = candidates[0][0]

        for s in range(top_k):
            x = candidates[s][1].copy()
            fx = candidates[s][0]
            sigma = 1.5  # Initial step size for sharp ridge
            
            # Track success rate for adaptive sigma
            successes = 0
            total = 0
            
            # (1+1)-ES: 1 offspring per generation
            for gen in range(budget_per_start):
                # Check deadline
                if time.time() > deadline:
                    break
                
                # Check budget
                if calls_made >= budget:
                    break

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                
                total += 1
                if fy < fx:
                    successes += 1
                    x = y
                    fx = fy
                    # Update global best
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                
                # Adaptive sigma based on success rate (target ~20-25%)
                if total >= 4:
                    success_rate = successes / total
                    if success_rate > 0.25:
                        sigma *= 1.2
                    elif success_rate < 0.15:
                        sigma *= 0.85
                    # Keep success window small for responsiveness
                    successes = max(0, successes - 2)
                    total = max(0, total - 4)
                
                # Ensure sigma stays reasonable
                sigma = np.clip(sigma, 0.01, 10.0)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
