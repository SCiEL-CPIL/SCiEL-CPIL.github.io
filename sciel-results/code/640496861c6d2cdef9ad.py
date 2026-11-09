import numpy as np
import time
import sys

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
        self._deadline = self._start_time + 320
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

        # Generate Sobol sequence for global exploration points
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(500)
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            sobol_points = self._rng.uniform(-5, 5, (500, self.dim))

        sobol_idx = 0
        total_sobol = len(sobol_points)

        # Phase 1: Quick global exploration with 20 Sobol points
        for i in range(20):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            x = sobol_points[sobol_idx]
            sobol_idx += 1
            safe_call(x)

        # Phase 2: CMA-ES with periodic global restarts
        # Use scipy's CMA-ES for better convergence, with Sobol-based restarts
        # when the optimizer stagnates
        
        from scipy.optimize import minimize
        
        # CMA-ES parameters
        maxiter = 200
        popsize = 12  # lambda = 12 for dim=5
        sigma0 = 1.5
        tol = 1e-8
        
        # We'll run CMA-ES in segments, restarting when stuck
        segment_budget = 80  # calls per CMA-ES segment
        num_segments = 5
        segment_calls = 0
        segment_stagnation = 0
        last_best_f = self._best_f
        
        for seg in range(num_segments):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            
            # Initialize CMA-ES at current best
            x0 = self._best_x.copy()
            
            # Track calls within this segment
            seg_start_calls = self._call_count
            seg_start_best = self._best_f
            
            # Define objective for CMA-ES
            def cma_objective(x):
                if self._call_count >= self.budget or time.time() >= self._deadline:
                    return 1e10
                x_clipped = np.clip(x, -5.0, 5.0)
                return safe_call(x_clipped)
            
            # Run CMA-ES
            try:
                result = minimize(
                    cma_objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'maxiter': 50,
                        'maxfev': 60
                    }
                )
            except Exception:
                pass
            
            # Check if we made progress
            if self._best_f < seg_start_best - 1e-6:
                # Made progress, reset stagnation
                segment_stagnation = 0
                last_best_f = self._best_f
            else:
                # No progress, consider restart
                segment_stagnation += 1
                
                # If stagnating, do a Sobol jump
                if segment_stagnation >= 2 and sobol_idx < total_sobol:
                    jump_x = sobol_points[sobol_idx]
                    sobol_idx += 1
                    f_jump = safe_call(jump_x)
                    if f_jump < self._best_f:
                        self._best_f = f_jump
                        self._best_x = jump_x
                        segment_stagnation = 0
                        last_best_f = self._best_f
                elif segment_stagnation >= 3:
                    # Larger random restart
                    restart_x = self._rng.uniform(-5, 5, self.dim)
                    f_restart = safe_call(restart_x)
                    if f_restart < self._best_f:
                        self._best_f = f_restart
                        self._best_x = restart_x
                        segment_stagnation = 0
                        last_best_f = self._best_f
            
            # Interleave with Sobol points for diversity
            if sobol_idx < total_sobol and self._call_count < self.budget:
                sobol_jump = sobol_points[sobol_idx]
                sobol_idx += 1
                f_sobol = safe_call(sobol_jump)
                if f_sobol < self._best_f:
                    self._best_f = f_sobol
                    self._best_x = sobol_jump
            
            if time.time() >= self._deadline - 10:
                break
        
        # Final refinement with (1+1)-ES on the best point
        sigma = 0.5
        sigma_min = 1e-8
        stagnation_counter = 0
        stagnation_limit = 15
        
        while self._call_count < self.budget and time.time() < self._deadline:
            if stagnation_counter >= stagnation_limit:
                # Try a Sobol or random restart
                if sobol_idx < total_sobol:
                    jump_x = sobol_points[sobol_idx]
                    sobol_idx += 1
                else:
                    jump_x = self._rng.uniform(-5, 5, self.dim)
                f_jump = safe_call(jump_x)
                if f_jump < self._best_f:
                    self._best_f = f_jump
                    self._best_x = jump_x
                    sigma = 0.5
                    stagnation_counter = 0
                else:
                    sigma = min(sigma * 2.0, 3.0)
                    stagnation_counter = 0
            
            noise = self._rng.standard_normal(self.dim)
            candidate_x = self._best_x + sigma * noise
            candidate_x = np.clip(candidate_x, -5.0, 5.0)
            
            f_cand = safe_call(candidate_x)
            
            if f_cand < self._best_f:
                self._best_f = f_cand
                self._best_x = candidate_x
                stagnation_counter = 0
                sigma = min(sigma * 1.05, 2.0)
            else:
                stagnation_counter += 1
                sigma = max(sigma * 0.9, sigma_min)
        
        return self._best_x, self._best_f
