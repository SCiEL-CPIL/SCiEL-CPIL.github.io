"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer for linear slope functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: finish well before 300s
        deadline = start_time + 290.0

        rng = np.random.default_rng(0)
        
        # Phase 1: Estimate Gradient with large steps for robustness
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        best_f = f0
        best_x = x0.copy()
        
        calls = 1
        
        # Large epsilon for robust gradient estimation
        eps = 2.0
        
        grad = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
                
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            fp = func(x_plus)
            fm = func(x_minus)
            calls += 2
            
            if fp < best_f:
                best_f = fp
                best_x = x_plus.copy()
            if fm < best_f:
                best_f = fm
                best_x = x_minus.copy()
                
            grad[i] = (fp - fm) / (2.0 * eps)
            
        # Phase 2: Jump to predicted optimal vertex
        x_vertex = np.zeros(self.dim)
        for i in range(self.dim):
            if grad[i] > 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 5.0
                
        f_vertex = func(x_vertex)
        calls += 1
        
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()
            
        # Phase 3: Aggressive local search with adaptive sigma
        x_curr = best_x.copy()
        f_curr = best_f
        
        # Start with moderate sigma, adapt based on progress
        sigma = 1.0
        no_improve_count = 0
        max_no_improve = 50
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate candidate
            y = x_curr + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5.0, 5.0)
            
            fy = func(y)
            calls += 1
            
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                x_curr = y.copy()
                f_curr = fy
                no_improve_count = 0
                # Increase sigma on success to explore more
                sigma = min(sigma * 1.15, 5.0)
            else:
                no_improve_count += 1
                sigma *= 0.85
                if sigma < 1e-8:
                    break
                
                # If stuck for a while, try a larger jump to escape local region
                if no_improve_count >= max_no_improve:
                    # Try a random restart from a different point
                    x_restart = rng.uniform(-5.0, 5.0, size=self.dim)
                    fr = func(x_restart)
                    calls += 1
                    if fr < best_f:
                        best_f = fr
                        best_x = x_restart.copy()
                        x_curr = x_restart.copy()
                        f_curr = fr
                        sigma = 1.0
                        no_improve_count = 0
                    else:
                        # Reduce sigma more aggressively
                        sigma = max(sigma * 0.5, 1e-9)
                        no_improve_count = 0
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
