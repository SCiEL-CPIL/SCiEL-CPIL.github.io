"""Simulated Annealing for BBOB f7 (Step ellipsoidal) with plateau-aware adaptation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with plateau-aware step adaptation for step ellipsoidal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.time()

        # Initial broad exploration with a few random points to find a good starting point
        n_initial = min(10, budget)
        for i in range(n_initial):
            if calls_made >= budget or time_left() < 5.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Main search: Simulated Annealing
        if best_x is not None and calls_made < budget and time_left() > 5.0:
            x = best_x.copy()
            fx = best_f
            
            # SA Parameters
            # Initial temperature high enough to accept most moves
            T_initial = 10.0
            # Final temperature low enough to converge
            T_final = 1e-6
            # Step size for perturbations
            # We can adapt this, but for simplicity, we start with a reasonable value
            # Box is [-5, 5], so range is 10. A step of 1.0 is 10% of range.
            step_size = 1.0
            
            # Number of SA steps
            # We have remaining budget. Let's use most of it for SA.
            # Reserve a few calls for a final local polish if needed, but SA usually converges well.
            sa_steps = budget - calls_made - 5 
            
            # Cooling schedule: Geometric
            # T_k = T_initial * (T_final / T_initial)^(k / sa_steps)
            cooling_rate = (T_final / T_initial) ** (1.0 / sa_steps)
            T = T_initial
            
            for k in range(sa_steps):
                if calls_made >= budget or time_left() < 2.0:
                    break
                
                # Generate neighbor
                y = x + step_size * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                # Acceptance criterion
                if fy < fx:
                    # Always accept improvement
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Accept worse solution with probability p
                    delta_f = fy - fx
                    if delta_f == 0:
                        # Plateau: accept with high probability to explore
                        p = 0.5
                    else:
                        # Avoid overflow in exp
                        if T < 1e-12:
                            p = 0.0
                        else:
                            p = np.exp(-delta_f / T)
                    
                    if rng.random() < p:
                        x = y
                        fx = fy
                        # Note: We do NOT update best_x/best_f here if it's worse
                        # But we do move the current state x to the worse solution to escape local optima
        
                # Cool down
                T *= cooling_rate
                
                # Adaptive step size? 
                # If we are stuck (fx == best_f for a long time), maybe increase step size?
                # Or just rely on SA's ability to jump.
                # Let's keep step size fixed for now to avoid complexity, 
                # but maybe reduce it slightly as T drops to refine.
                if T < 1.0:
                    step_size *= 0.95
                elif T < 0.1:
                    step_size *= 0.9

        # Final polish: fine-grained local search around best point
        if best_x is not None and calls_made < budget and time_left() > 2.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_steps = min(10, budget - calls_made)
            for _ in range(polish_steps):
                if calls_made >= budget or time_left() < 1.0:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
