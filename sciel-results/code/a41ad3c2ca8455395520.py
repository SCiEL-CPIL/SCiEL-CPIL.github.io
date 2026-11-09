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
        # 330s limit, keep 5s margin
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

        # Generate Sobol sequence for global points
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(500)
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            sobol_points = self._rng.uniform(-5, 5, (500, self.dim))

        # Phase 1: Global Exploration with Sobol points
        sobol_idx = 0
        
        # Evaluate first 30 Sobol points to find a decent basin
        for i in range(30):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            x = sobol_points[sobol_idx]
            sobol_idx += 1
            f = safe_call(x)

        # Phase 2: (1+1)-ES with CSA-like step size adaptation and Restarts
        # Use cumulative step-size adaptation for more robust convergence
        sigma = 1.0
        sigma_min = 1e-7
        sigma_max = 5.0
        stagnation_limit = 40
        stagnation_counter = 0
        
        # CSA parameters
        c_sigma = 0.8  # adaptation rate
        d_sigma = 1.5  # damping
        chi = (self.dim ** 0.5) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))  # expected ||N(0,I)||
        
        # Track cumulative step for CSA
        p_sigma = np.zeros(self.dim)
        
        # We will loop until budget or time is out
        while self._call_count < self.budget and time.time() < self._deadline:
            # Generate candidate
            noise = self._rng.standard_normal(self.dim)
            candidate_x = self._best_x + sigma * noise
            
            # Clip to box [-5, 5]
            candidate_x = np.clip(candidate_x, -5.0, 5.0)
            
            f_cand = safe_call(candidate_x)
            
            if f_cand < self._best_f:
                # Accept
                self._best_f = f_cand
                self._best_x = candidate_x
                stagnation_counter = 0
                # CSA: adapt sigma based on actual step length
                step = candidate_x - (self._best_x - sigma * noise)  # approximate
                actual_step = np.linalg.norm(noise)
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (actual_step / chi - 1)
                sigma *= np.exp(c_sigma * d_sigma * p_sigma.mean())
                sigma = np.clip(sigma, sigma_min, sigma_max)
            else:
                # Reject
                stagnation_counter += 1
                # CSA: adapt sigma
                actual_step = np.linalg.norm(noise)
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (actual_step / chi - 1)
                sigma *= np.exp(c_sigma * d_sigma * p_sigma.mean())
                sigma = np.clip(sigma, sigma_min, sigma_max)
                
                # Check for stagnation
                if stagnation_counter >= stagnation_limit:
                    # Restart: Pick a new point from Sobol sequence or random
                    if sobol_idx < len(sobol_points):
                        new_x = sobol_points[sobol_idx]
                        sobol_idx += 1
                    else:
                        new_x = self._rng.uniform(-5, 5, self.dim)
                    
                    f_new = safe_call(new_x)
                    
                    # Update best if the new global point is better
                    if f_new < self._best_f:
                        self._best_f = f_new
                        self._best_x = new_x
                        # Reset sigma to a medium value for new basin
                        sigma = 1.0
                        p_sigma = np.zeros(self.dim)
                        stagnation_counter = 0
                    else:
                        # Keep current best but reset stagnation counter
                        # Increase sigma to jump out of local minimum
                        sigma = min(sigma * 2.5, sigma_max)
                        p_sigma = np.zeros(self.dim)
                        stagnation_counter = 0

        # Phase 3: Final high-precision polish pass
        # Use smaller sigma for fine-tuning the best solution
        polish_budget = min(50, max(10, (self.budget - self._call_count) // 4))
        if self._call_count < self.budget and time.time() < self._deadline:
            sigma_polish = max(sigma * 0.1, 1e-4)
            for i in range(polish_budget):
                if self._call_count >= self.budget or time.time() >= self._deadline:
                    break
                noise = self._rng.standard_normal(self.dim)
                candidate_x = self._best_x + sigma_polish * noise
                candidate_x = np.clip(candidate_x, -5.0, 5.0)
                f_cand = safe_call(candidate_x)
                if f_cand < self._best_f:
                    self._best_f = f_cand
                    self._best_x = candidate_x
                    sigma_polish *= 0.95  # gradually reduce step size
                else:
                    sigma_polish *= 0.98

        return self._best_x, self._best_f
