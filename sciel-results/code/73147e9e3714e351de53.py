"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Two-stage optimizer: Sobol grid search + Multi-restart (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        t0 = time.time()
        deadline = t0 + 280.0

        calls = 0
        best_x = None
        best_f = float('inf')

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: Sobol Grid Search ---
        # Use Sobol sequence for low-discrepancy sampling
        n_phase1 = 100
        sampler = qmc.Sobol(d, scramble=True)
        sobol_samples = sampler.random(n_phase1)
        # Map [0, 1]^d to [-5, 5]^d
        sobol_samples = sobol_samples * 10.0 - 5.0
        
        candidates = []
        for i in range(n_phase1):
            if calls >= budget or time.time() > deadline:
                break
            x = clip(sobol_samples[i])
            fx = float(func(x))
            calls += 1
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates and keep top K for Phase 2
        candidates.sort(key=lambda item: item[0])
        top_k = candidates[:5]
        
        # --- Phase 2: Multi-restart (1+1)-ES ---
        # Allocate remaining budget to top candidates
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f
            
        n_restarts = len(top_k)
        budget_per_restart = remaining_budget // n_restarts
        if budget_per_restart < 10:
            budget_per_restart = 10
            n_restarts = max(1, remaining_budget // budget_per_restart)

        for k in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Start from the k-th best candidate
            start_x = top_k[k][1].copy()
            start_f = top_k[k][0]
            
            # (1+1)-ES parameters
            sigma = 1.0
            x = start_x
            fx = start_f
            
            for step in range(budget_per_restart):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate candidate
                y = clip(x + sigma * rng.standard_normal(d))
                fy = float(func(y))
                calls += 1
                
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.25
                else:
                    sigma *= 0.85
                    # Optional: if sigma becomes too small, restart or break
                    if sigma < 1e-4:
                        break
                
                sigma = min(sigma, 4.0)
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
