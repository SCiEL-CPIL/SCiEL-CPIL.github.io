"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer with adaptive (1+1)-ES and restarts for multi-modal search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid search to find promising regions
        # Use Latin Hypercube Sampling for better coverage
        n_lhs = min(50, self.budget // 10)
        lhs_points = []
        for i in range(n_lhs):
            x = (np.arange(self.dim) / self.dim + rng.random()) / self.dim * 10 - 5
            lhs_points.append(x)
        
        for x in lhs_points:
            if calls_made >= self.budget or time_left() < 5:
                break
            evaluate(x)

        # Phase 2: Local refinement from best points using (1+1)-ES
        # Sort by fitness and take top candidates for local search
        candidates = []
        if best_x is not None:
            candidates.append((best_f, best_x.copy()))

        # Add some random points near the best
        for _ in range(min(5, max(1, self.budget // 20))):
            if best_x is not None and time_left() > 10:
                perturbation = rng.standard_normal(self.dim) * 0.5
                x_new = np.clip(best_x + perturbation, -5, 5)
                fx = evaluate(x_new)
                if fx < best_f * 2 + 100:  # Reasonable threshold
                    candidates.append((fx, x_new.copy()))

        # Local search from each candidate
        for init_f, init_x in candidates:
            if calls_made >= self.budget or time_left() < 10:
                break
            
            x = init_x.copy()
            fx = init_f
            sigma = 1.0
            no_improve = 0
            max_no_improve = 100
            
            while calls_made < self.budget and time_left() > 2:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break

        # Phase 3: Final polish with smaller steps
        if calls_made < self.budget and time_left() > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            no_improve = 0
            
            while calls_made < self.budget and time_left() > 1:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1
                    if no_improve > 50:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
