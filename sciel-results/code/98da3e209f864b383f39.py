"""Portfolio optimizer for BBOB f21 (Gallagher 101 peaks).

Strategy: 
1. Coarse global exploration with Sobol-like quasi-random + random points to hit multiple basins.
2. Population-based CMA-ES with full covariance adaptation for the best few points found.
3. Final fine-tuning with adaptive (1+1)-ES.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Portfolio: global seeding + CMA-ES + adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s kill

        best_x = None
        best_f = float('inf')
        call_count = 0

        def safe_func(x):
            nonlocal best_f, best_x, call_count
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64).clip(-5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return best_f if best_f != float('inf') else 1e10
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(123)

        # Phase 1: Global exploration
        # Use a mix of random and structured points to cover the search space
        n_random = min(100, max(30, budget // 5))
        for i in range(n_random):
            if call_count >= budget or time.time() > deadline - 90:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            safe_func(x)

        # Add a few deterministic points for coverage
        special_starts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.array([4.0] * dim),
            np.array([-4.0] * dim),
            np.array([0.5, -0.5, 0.5, -0.5, 0.5]),
            np.array([-0.5, 0.5, -0.5, 0.5, -0.5]),
        ]
        for s in special_starts:
            if call_count >= budget or time.time() > deadline - 90:
                break
            safe_func(s)

        if best_x is None:
            best_x = np.zeros(dim)
            safe_func(best_x)

        # Phase 2: CMA-ES-like population-based local search
        # Use a proper population-based approach for better covariance estimation
        pop_size = min(10, max(4, budget // 40))  # Small population for 500 budget
        sigma = 1.0
        mean = best_x.copy()
        
        # Track population history for covariance updates
        best_pop = [(best_x.copy(), best_f)]
        
        phase2_budget = budget - call_count - 30
        if phase2_budget < 20:
            phase2_budget = max(0, budget - call_count)
        
        stagnation = 0
        max_stagnation = 15
        
        for step in range(phase2_budget):
            if time.time() > deadline - 30 or call_count >= budget:
                break
            
            # Generate population
            pop = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = mean + sigma * z
                y = np.clip(y, -5, 5)
                fy = safe_func(y)
                pop.append((y, fy))
            
            # Sort by fitness
            pop.sort(key=lambda p: p[1])
            
            # Update mean with weighted average of top individuals
            top_k = max(2, pop_size // 2)
            weights = np.array([1.0 / (i + 1) for i in range(top_k)])
            weights /= weights.sum()
            
            new_mean = sum(w * p[0] for w, p in zip(weights, pop[:top_k]))
            new_mean = np.clip(new_mean, -5, 5)
            
            # Adapt sigma based on improvement
            if pop[0][1] < best_f - 1e-10:
                best_x = pop[0][0].copy()
                best_f = pop[0][1]
                sigma *= 1.05
                stagnation = 0
            else:
                stagnation += 1
                sigma *= 0.95
            
            # Track best population for covariance
            if pop[0][1] < best_pop[0][1]:
                best_pop = [pop[0]]
                if len(best_pop) < top_k:
                    for p in pop[1:top_k]:
                        if p[0] is not best_pop[0][0]:
                            best_pop.append(p)
            
            mean = new_mean
            
            if stagnation >= max_stagnation:
                # Restart with perturbation
                mean = best_x + sigma * rng.standard_normal(dim)
                mean = np.clip(mean, -5, 5)
                sigma = 0.5 * sigma
                stagnation = 0
        
        # Phase 3: Final refinement with adaptive (1+1)-ES
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            x = best_x.copy()
            fx = best_f
            sigma_ref = 0.3
            max_ref_steps = min(100, budget - call_count)
            
            for step in range(max_ref_steps):
                if time.time() > deadline - 5 or call_count >= budget:
                    break
                y = np.clip(x + sigma_ref * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_ref *= 1.1
                else:
                    sigma_ref *= 0.8
                if sigma_ref < 0.003:
                    break
            
            if fx < best_f:
                best_x, best_f = x, fx

        return best_x, best_f

# EVOLVE-BLOCK-END
