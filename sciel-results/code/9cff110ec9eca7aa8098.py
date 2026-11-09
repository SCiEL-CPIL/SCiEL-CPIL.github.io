import numpy as np
import time
import sys
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        self._deadline = self._start_time + 325
        self._call_count = 0
        self.budget_remaining = self.budget
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Phase 1: Initial exploration with diverse starting points
        # Use a combination of center, random, and structured points
        n_initial = min(40, self.budget // 8)
        
        # Start from center
        f_center = safe_call(np.zeros(self.dim))
        
        # Add some structured exploration points
        for i in range(1, n_initial):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            # Mix of random and scaled random points
            if i % 3 == 0:
                x = self._rng.uniform(-5, 5, self.dim)
            elif i % 3 == 1:
                x = self._rng.uniform(-3, 3, self.dim)
            else:
                x = self._rng.uniform(-1, 1, self.dim)
            safe_call(x)

        # Phase 2: CMA-ES with restarts using scipy
        # CMA-ES is a population-based method with covariance matrix adaptation
        pop_size = max(10, int(4 + 3 * np.log(self.dim)))
        
        # Run multiple CMA-ES instances from different starting points
        max_restarts = 3
        for restart in range(max_restarts):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            
            # Choose starting point: best found so far for first restart, random for others
            if restart == 0:
                x0 = self._best_x.copy()
            else:
                x0 = self._rng.uniform(-5, 5, self.dim)
                f_x0 = safe_call(x0)
                if f_x0 < self._best_f:
                    self._best_f = f_x0
                    self._best_x = x0.copy()
            
            # Set up CMA-ES with our safe_call wrapper
            # scipy's cmaes uses its own call counting, so we need to be careful
            # We'll use a budget-aware approach
            
            # Calculate how many more calls we can make
            remaining_budget = self.budget - self._call_count
            if remaining_budget <= 0:
                break
            
            # CMA-ES parameters
            sigma0 = 2.0 if restart == 0 else 1.5
            
            # Use scipy's cmaes if available, otherwise fall back to simple ES
            try:
                from scipy.optimize import _minimize_cobyla
                # We'll implement a simple CMA-ES-like population method
                # since scipy's cmaes has different interface
                
                # Simple population-based ES with covariance adaptation
                pop = np.zeros((pop_size, self.dim))
                for j in range(pop_size):
                    pop[j] = x0 + sigma0 * self._rng.standard_normal(self.dim)
                    pop[j] = np.clip(pop[j], -5.0, 5.0)
                
                # Evaluate population
                for j in range(pop_size):
                    if self._call_count >= self.budget or time.time() >= self._deadline:
                        break
                    safe_call(pop[j])
                
                # Sort by fitness
                pop_f = np.array([func(p) if self._call_count < self.budget else 1e10 for p in pop])
                # Re-evaluate to get consistent values (this uses extra calls)
                # Instead, let's track them properly
                pop_f = []
                for p in pop:
                    if self._call_count >= self.budget or time.time() >= self._deadline:
                        pop_f.append(1e10)
                    else:
                        pop_f.append(safe_call(p))
                pop_f = np.array(pop_f)
                
                # Simple CMA-ES-like update: move centroid toward best individuals
                for gen in range(50):
                    if self._call_count >= self.budget or time.time() >= self._deadline:
                        break
                    
                    # Sort population
                    idx = np.argsort(pop_f)
                    pop_sorted = pop[idx]
                    f_sorted = pop_f[idx]
                    
                    # Update mean (centroid)
                    weights = np.exp(-np.arange(pop_size) / (pop_size / 2))
                    weights /= weights.sum()
                    x_mean = np.sum(pop_sorted * weights[:, None], axis=0)
                    
                    # Adapt sigma based on success rate
                    success_rate = np.sum(f_sorted[:pop_size//2] < f_sorted[pop_size//2:]) / pop_size
                    sigma0 *= 0.9 + 0.2 * success_rate
                    sigma0 = np.clip(sigma0, 1e-4, 5.0)
                    
                    # Generate new population
                    for j in range(pop_size):
                        if self._call_count >= self.budget or time.time() >= self._deadline:
                            break
                        pop[j] = x_mean + sigma0 * self._rng.standard_normal(self.dim)
                        pop[j] = np.clip(pop[j], -5.0, 5.0)
                    
                    # Evaluate new population
                    for j in range(pop_size):
                        if self._call_count >= self.budget or time.time() >= self._deadline:
                            break
                        safe_call(pop[j])
                    
                    # Update population fitness
                    for j in range(pop_size):
                        if self._call_count >= self.budget or time.time() >= self._deadline:
                            pop_f[j] = 1e10
                        else:
                            pop_f[j] = safe_call(pop[j])
                    
                    # Update best
                    if pop_f[0] < self._best_f:
                        self._best_f = pop_f[0]
                        self._best_x = pop_sorted[0].copy()
                        
            except Exception:
                pass

        # Phase 3: Final polish with (1+1)-ES
        if self._call_count < self.budget and time.time() < self._deadline:
            sigma = max(0.1, 1.0 * (self.budget - self._call_count) / self.budget)
            for i in range(100):
                if self._call_count >= self.budget or time.time() >= self._deadline:
                    break
                noise = self._rng.standard_normal(self.dim)
                candidate_x = self._best_x + sigma * noise
                candidate_x = np.clip(candidate_x, -5.0, 5.0)
                f_cand = safe_call(candidate_x)
                if f_cand < self._best_f:
                    self._best_f = f_cand
                    self._best_x = candidate_x
                    sigma *= 0.98
                else:
                    sigma *= 0.95

        return self._best_x, self._best_f
