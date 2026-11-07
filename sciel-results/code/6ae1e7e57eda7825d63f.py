"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive optimizer with deadline guard for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        def remaining_time():
            return deadline - time.time()
        
        # Phase 1: Multi-start exploration
        best_x = None
        best_f = np.inf
        n_starts = min(8, self.budget // 50)
        
        for i in range(n_starts):
            if time.time() > deadline - 5:
                break
            # Diverse starting points
            if i == 0:
                x = np.zeros(self.dim)
            elif i == 1:
                x = rng.uniform(-4, 4, self.dim)
            else:
                # Mix of random and structured
                x = rng.uniform(-5, 5, self.dim)
                if i % 2 == 0:
                    x = np.clip(x * 0.7, -5, 5)
            
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
        
        # Phase 2: Adaptive local search from best point
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        stagnation = 0
        max_stagnation = 30
        total_calls = n_starts
        min_sigma = 1e-4
        max_sigma = 5.0
        
        while total_calls < self.budget and remaining_time() > 1.0:
            # Check time more frequently
            if total_calls % 10 == 0 and time.time() > deadline - 2:
                break
            
            # Adaptive step size based on recent success rate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            total_calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                stagnation = 0
                sigma = min(sigma * 1.3, max_sigma)
            else:
                stagnation += 1
                sigma *= 0.85
                
                # Restart if stuck too long
                if stagnation >= max_stagnation:
                    # Try a perturbation from best point
                    if remaining_time() > 5.0:
                        x = best_x + 2.0 * rng.standard_normal(self.dim)
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        total_calls += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma = 1.0
                        stagnation = 0
                        max_stagnation = min(max_stagnation * 1.2, 100)
                    else:
                        # Just shrink sigma aggressively
                        sigma *= 0.5
                        stagnation = 0
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # If sigma is too small, try a larger jump
            if sigma < min_sigma:
                if remaining_time() > 5.0:
                    x = best_x + 3.0 * rng.standard_normal(self.dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    total_calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = 1.0
                    stagnation = 0
                else:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
