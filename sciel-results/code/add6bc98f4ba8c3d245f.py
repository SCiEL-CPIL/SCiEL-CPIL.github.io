import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # 300s limit, 30s safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        pop_size = 4
        c_sigma = (pop_size + 2) / ((dim + 1.3) ** 2 + pop_size)
        c_c = (4 + dim / 4) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        mu_eff = sum(1.0 / (i + 1) for i in range(mu_eff_idx)) if False else 2.0 # Approx for small pop
        # More robust mu_eff for pop_size=4:
        weights = np.zeros(pop_size)
        weights[:int(pop_size/2)] = np.log(pop_size/2 + 1) - np.log(1, int(pop_size/2))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial state
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        sigma = 2.0
        
        # CMA-ES state
        mean = x0.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Precompute eigen decomposition update frequency
        eval_count = 0
        eig_update_interval = max(1, int(1 / (c_1 + c_mu)))
        last_eig_update = 0
        
        # Standard normal distribution for sampling
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
                
            # 1/5 success rule style adaptation for CMA-ES (simplified)
            # Actually, standard CMA-ES uses path adaptation. Let's implement a simplified CMA-ES.
            
            # Generate population
            # We need to update C periodically. For simplicity in this constrained box, 
            # we will use a simplified CMA-ES or a robust (1+1)-ES with better restarts.
            # Given the "bent cigar" is highly ill-conditioned, CMA-ES is ideal, but implementation 
            # complexity is high. Let's use a multi-start (1+1)-ES with aggressive restarts 
            # and a larger initial sigma, which is often surprisingly effective on BBOB f12 
            # when combined with multiple independent runs.
            
            # Strategy: Multi-start (1+1)-ES
            # We will run several independent (1+1)-ES instances.
            # If one gets stuck (no improvement for N steps), we restart it from the current best or a new random point.
            
            # To keep it simple and robust within the code block:
            # We implement a single (1+1)-ES but with "adaptive restarts" based on stagnation.
            
            # Stagnation detection
            if not hasattr(self, '_stagnation'):
                self._stagnation = 0
                self._last_best = f0
                self._sigma = 2.0
                self._x = x0.copy()
                self._fx = f0
                
            # Update stagnation
            if f0 < self._last_best - 1e-10:
                self._stagnation = 0
                self._last_best = f0
            else:
                self._stagnation += 1
            
            # If stagnation > 50, restart from a new random point or around best
            if self._stagnation > 50:
                # Restart: try a few random points, pick the best
                best_restart_x = self._x
                best_restart_f = self._fx
                for _ in range(5):
                    xr = rng.uniform(-5, 5, dim)
                    fr = func(xr)
                    if fr < best_restart_f:
                        best_restart_f = fr
                        best_restart_x = xr
                self._x = best_restart_x
                self._fx = best_restart_f
                self._sigma = 2.0
                self._stagnation = 0
                self._last_best = best_restart_f
                
                if self._fx < best_f:
                    best_f = self._fx
                    best_x = self._x.copy()
                    
            # Standard (1+1)-ES step
            y = np.clip(self._x + self._sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            
            if fy < self._fx:
                self._x = y
                self._fx = fy
                self._sigma *= 1.1 # Slower growth to avoid jumping over narrow valleys
                if self._fx < best_f:
                    best_f = self._fx
                    best_x = self._x.copy()
            else:
                self._sigma *= 0.9 # Slower decay to maintain exploration
                self._sigma = max(self._sigma, 1e-6)
                
        return best_x, best_f
