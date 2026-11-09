import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 285  # safety margin before 300s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        deadline = self.deadline

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(12345)

        # Phase 1: Diverse initial sampling (20 points)
        n_init = 20
        candidates = []
        # Include center and some structured points
        points = [np.zeros(dim)]
        # Add corners of a smaller box
        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                for k in range(-1, 2, 2):
                    p = np.array([i*3, j*3, k*3, 0, 0], dtype=float)
                    points.append(p)
        # Fill rest with random
        while len(points) < n_init:
            points.append(rng.uniform(-5, 5, dim))

        for p in points:
            if calls >= budget or time.time() > deadline:
                break
            f_val = eval_func(p)
            candidates.append((f_val, p.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])

        # Phase 2: Multi-start local optimization with (1+1)-ES
        # Use the best few candidates as starting points for local search
        # We'll use a simple adaptive step-size (1+1)-ES for each start
        
        # Allocate remaining budget among up to 3 best starts
        n_starts = min(3, len(candidates))
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Distribute budget: give more to better starts
        # Simple heuristic: 50% to best, 30% to second, 20% to third
        if n_starts == 1:
            budgets_per_start = [remaining_budget]
        elif n_starts == 2:
            budgets_per_start = [int(0.6 * remaining_budget), remaining_budget - int(0.6 * remaining_budget)]
        else:
            b1 = int(0.5 * remaining_budget)
            b2 = int(0.3 * remaining_budget)
            b3 = remaining_budget - b1 - b2
            budgets_per_start = [b1, b2, b3]

        for i in range(n_starts):
            if calls >= budget or time.time() > deadline:
                break
            if budgets_per_start[i] <= 0:
                continue
            
            x = candidates[i][1].copy()
            f = candidates[i][0]
            
            # (1+1)-ES with adaptive step size
            step_size = 1.0  # initial step
            min_step = 1e-4
            max_step = 10.0
            no_improve_count = 0
            max_no_improve = 10
            
            while calls < budget and time.time() < deadline:
                # Check if we've exhausted our allocated budget for this start
                if calls >= budget:
                    break
                
                # Generate a mutant
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-15)
                x_new = x + step_size * direction
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = eval_func(x_new)
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    no_improve_count = 0
                    # Increase step size slightly on success
                    step_size = min(step_size * 1.2, max_step)
                else:
                    no_improve_count += 1
                    # Decrease step size on failure
                    step_size = max(step_size * 0.8, min_step)
                    
                    if no_improve_count >= max_no_improve:
                        # Restart local search from current best with smaller step
                        step_size = max(step_size * 0.5, min_step)
                        no_improve_count = 0
                        if step_size <= min_step:
                            break

        return best_x, best_f
