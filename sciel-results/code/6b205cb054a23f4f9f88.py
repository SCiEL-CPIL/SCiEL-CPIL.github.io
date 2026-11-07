import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        calls = 0
        
        # Phase 1: Global Sampling
        # Use a diverse set of points
        n_starts = 12
        points = []
        # Center
        points.append(np.zeros(dim))
        # Corners (subset)
        points.append(np.full(dim, -4.0))
        points.append(np.full(dim, 4.0))
        points.append(np.full(dim, -2.0))
        points.append(np.full(dim, 2.0))
        
        # Random stratified
        for _ in range(n_starts - 5):
            # Use Halton-like or just random
            points.append(rng.uniform(-4.5, 4.5, dim))
            
        # Evaluate
        best_x = None
        best_f = np.inf
        for p in points:
            if calls >= budget or time.time() > deadline:
                break
            p = np.clip(p, -5, 5)
            f = func(p)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = p.copy()
                
        if best_x is None:
            # Fallback
            p = rng.uniform(-5, 5, dim)
            best_f = func(p)
            calls += 1
            best_x = p
            
        # Phase 2: CMA-ES Refinement
        if calls < budget and time.time() < deadline:
            best_x, best_f = self._cmaes_refine(func, best_x, best_f, budget - calls, rng, deadline)
            
        return best_x, best_f

    def _cmaes_refine(self, func, x, fx, budget_left, rng, deadline):
        dim = self.dim
        # CMA-ES Parameters
        lam = 6  # Population size
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        # Initial state
        x_mean = x.copy()
        sigma = 0.5  # Start with smaller sigma as we are in a local basin
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 3)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        calls_used = 0
        
        while calls_used < budget_left and time.time() < deadline:
            # Check if we can afford a generation
            if calls_used + lam > budget_left:
                # Use remaining budget for single steps or break
                # For simplicity, if we can't do a full gen, do random perturbation of best
                while calls_used < budget_left and time.time() < deadline:
                    y = x_mean + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        fx = fy
                        x_mean = y.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                    sigma = np.clip(sigma, 1e-5, 5.0)
                break
                
            # Generate offspring
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append((y, fy))
                
                # Early update of best
                if fy < fx:
                    fx = fy
                    # We don't update x_mean here, wait for recombination
                    # But we can store it? No, CMA-ES updates mean via recombination.
            
            # Sort
            offspring.sort(key=lambda t: t[1])
            
            # Recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * offspring[i][0]
            
            # Update CMA-ES state
            # ... (Standard CMA-ES update logic)
            
            x_mean = x_new
            
        return x_mean, fx
