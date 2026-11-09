"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with geometric cooling for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls = 1
        
        # Initial temperature: large enough to escape local minima
        # Rastrigin has local minima every 2 units, so step size ~2-4 is good
        T = 10.0
        T_min = 1e-6
        # Cooling rate: we have budget-1 steps
        # T * alpha^(budget-1) = T_min
        # alpha = (T_min / T) ** (1 / (budget - 1))
        n_steps = self.budget - 1
        alpha = (T_min / T) ** (1.0 / n_steps)
        
        # Current best
        best_x = x.copy()
        best_fx = fx
        
        for _ in range(n_steps):
            if time.time() > deadline:
                break
                
            # Generate perturbation scaled by temperature
            # Use a step size that adapts: larger when T is high
            # For Rastrigin, the periodicity is 2*pi in the original space,
            # but after rotation/shift, the structure is complex.
            # A Gaussian perturbation with sigma proportional to sqrt(T) or just T works.
            # Let's use sigma = T * 0.5 to keep steps reasonable.
            sigma = T * 0.5
            # Cap sigma to avoid huge jumps that are always rejected
            sigma = min(sigma, 5.0)
            
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            # Acceptance criterion
            if fy < fx:
                # Always accept improvement
                x, fx = y, fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Accept with probability exp(-(fy - fx) / T)
                delta = fy - fx
                # Avoid overflow in exp
                if delta / T < 50:  # exp(-50) is ~1.9e-22, effectively zero
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                
                if rng.random() < prob:
                    x, fx = y, fy
            
            # Cool down
            T *= alpha
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
