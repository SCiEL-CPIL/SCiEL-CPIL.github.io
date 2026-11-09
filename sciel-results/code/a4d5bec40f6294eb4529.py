"""Combined optimizer: adaptive global scan + multi-restart (1+1)-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive global scan followed by multi-restart (1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Phase 1: Global scan to find promising basins
        # Use 50 points to increase chance of hitting global basin in Gallagher 101
        n_scan = 50
        xs = rng.uniform(-5, 5, size=(n_scan, self.dim))
        fs = np.full(n_scan, np.inf)
        
        calls_made = 0
        for i in range(n_scan):
            if time.time() > deadline:
                break
            fx = func(xs[i])
            fs[i] = fx
            calls_made += 1
        
        if calls_made == 0:
            # Fallback if even scan fails (shouldn't happen)
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            return x, fx

        # Sort scan results
        sorted_idx = np.argsort(fs)
        
        # Select top K points for local refinement
        # With 500 budget, 50 used for scan, 450 remaining.
        # Use top 6 points for more diverse local searches.
        k = min(6, n_scan)
        top_idxs = sorted_idx[:k]
        
        best_x = xs[top_idxs[0]].copy()
        best_f = fs[top_idxs[0]]

        # Phase 2: Local refinement using (1+1)-ES on each top point
        # Run ES on each restart point with adaptive budget allocation.
        # Start with the most promising points and allocate more budget to
        # those that show progress.
        
        remaining_budget = self.budget - calls_made
        # Allocate budget proportionally: more to earlier (better) points
        # but ensure each gets a reasonable amount
        budgets_per_restart = []
        total_weight = sum(range(1, k + 1))  # weights: 1, 2, 3, ..., k
        for i in range(k):
            weight = (k - i)  # First point gets highest weight
            budgets_per_restart.append(int(remaining_budget * weight / total_weight))
        
        # Ensure we don't exceed remaining budget
        for i in range(k):
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = xs[top_idxs[i]].copy()
            fx = fs[top_idxs[i]]
            
            # (1+1)-ES parameters
            sigma = 1.0  # Start with moderate step size
            local_budget = budgets_per_restart[i]
            no_improve_count = 0
            max_no_improve = 15  # Stop if no improvement for 15 consecutive steps
            
            for _ in range(local_budget):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3  # Increase step size on success
                    no_improve_count = 0
                else:
                    sigma *= 0.85  # Decrease step size on failure
                    no_improve_count += 1
                    
                # Early stop if sigma becomes very small or no progress
                if sigma < 1e-7 or no_improve_count >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
