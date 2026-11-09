"""Rewritten optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global scan followed by multi-restart adaptive (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: finish well before 330s limit (evaluator kills at 330s)
        # Use 300s as internal deadline to be safe.
        deadline = start_time + 300.0 
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility of the search strategy, 
        # but the landscape is seeded by the evaluator.
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline - 5.0

        # Phase 1: Global Scan
        # For dim=5, a 3^5 = 243 grid is manageable if we sample a subset.
        # We want to cover [-5, 5]^5.
        # Let's use a Latin Hypercube-like or Stratified approach.
        n_scan = min(60, budget // 4) # Use ~25% of budget for exploration
        
        # Stratified sampling: divide each dimension into bins
        n_bins = 3
        # Generate all grid centers
        grid_coords = []
        for i in range(n_bins ** dim):
            x = np.zeros(dim)
            tmp = i
            for d in range(dim):
                bin_idx = tmp % n_bins
                tmp //= n_bins
                # Center of the bin in [0, 1)
                center = (bin_idx + 0.5) / n_bins
                # Add some jitter within the bin
                jitter = rng.uniform(0, 1 / n_bins)
                x[d] = center + jitter
            # Map [0, 1]^dim to [-5, 5]^dim
            x = x * 10.0 - 5.0
            grid_coords.append(x)
        
        # Shuffle and pick n_scan points
        indices = rng.permutation(len(grid_coords))[:n_scan]
        scan_points = [grid_coords[i] for i in indices]

        best_fx = float('inf')
        best_x = None
        candidates = []

        for x in scan_points:
            if not time_ok():
                break
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if not candidates:
            # Fallback if we ran out of time immediately
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            return x0, fx0

        # Sort candidates to get the best starting points for local search
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-restart (1+1)-ES
        # Allocate remaining budget to local refinements
        remaining_budget = budget - len(candidates)
        
        # How many restarts? 
        # Let's try restarting from the top K candidates.
        # K depends on remaining budget. Each restart needs ~20-50 evals to converge well.
        # Let's say we allow 3-5 restarts.
        n_restarts = min(5, len(candidates), max(1, remaining_budget // 30))
        
        # For each restart, we run an adaptive (1+1)-ES
        for r in range(n_restarts):
            if not time_ok() or remaining_budget <= 10:
                break
            
            start_x = candidates[r][1].copy()
            start_fx = candidates[r][0]
            
            # Adaptive (1+1)-ES
            x = start_x
            fx = start_fx
            
            # Initial step size: large enough to explore, small enough to converge
            # If we started from a scan point, we are in a basin.
            # Start with sigma ~ 0.5
            sigma = 0.5
            # Success rate tracking
            successes = 0
            total_steps = 0
            
            # Run until time/budget runs out or convergence
            while time_ok() and remaining_budget > 0:
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                # Clip to bounds
                y = np.clip(y, -5.0, 5.0)
                
                fy = func(y)
                remaining_budget -= 1
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    successes += 1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    # Failure
                    pass
                
                total_steps += 1
                
                # Adaptive rule: adjust sigma every 10 steps
                if total_steps % 10 == 0:
                    rate = successes / 10.0
                    # Heuristic: aim for ~1/3 success rate (optimal for 1+1 ES)
                    if rate > 0.35:
                        sigma *= 1.25
                    elif rate < 0.20:
                        sigma *= 0.8
                    successes = 0
                    
                    # Stagnation detection: if sigma is very small, restart or break
                    if sigma < 1e-4:
                        break

                # Early stopping if improvement is negligible
                if fx > 1000.0: # Unlikely for f21, but safe
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
