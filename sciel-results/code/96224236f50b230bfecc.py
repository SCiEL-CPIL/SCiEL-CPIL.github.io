import numpy as np
import time

class Optimizer:
    """Multi-start Hooke-Jeeves with adaptive step size for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def hooke_jees(start_x, initial_step=0.5, min_step=1e-7, max_iter=2000):
            nonlocal eval_count
            if eval_count >= budget:
                return

            x = np.clip(start_x, -5, 5).copy()
            f = eval_func(x)
            if eval_count >= budget:
                return

            step_sizes = np.ones(d) * initial_step
            min_step_val = min_step
            expand_factor = 2.0
            shrink_factor = 0.5

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                moved = False
                for i in range(d):
                    if eval_count >= budget or time.time() > deadline:
                        break

                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    f_plus = eval_func(x_plus)
                    if eval_count >= budget:
                        break

                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        step_sizes[i] *= expand_factor
                        step_sizes[i] = min(step_sizes[i], 10.0)
                        moved = True
                        continue
                    else:
                        step_sizes[i] *= shrink_factor

                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= step_sizes[i]
                    f_minus = eval_func(x_minus)
                    if eval_count >= budget:
                        break

                    if f_minus < f:
                        x = x_minus
                        f = f_minus
                        step_sizes[i] *= expand_factor
                        step_sizes[i] = min(step_sizes[i], 10.0)
                        moved = True
                    else:
                        step_sizes[i] *= shrink_factor

                if np.all(step_sizes < min_step_val):
                    break

        # Phase 1: Multi-start exploration
        rng = np.random.default_rng(42)
        
        # Start with origin
        hooke_jees(np.zeros(d), initial_step=1.0)
        
        # Try several random starts
        num_restarts = 8
        for i in range(num_restarts):
            if eval_count >= budget - 50 or time.time() > deadline:
                break
            
            # Generate random starting point
            x_rand = rng.uniform(-4, 4, d)
            hooke_jees(x_rand, initial_step=0.5)
        
        # Phase 2: Fine-tune around the best found
        if eval_count < budget - 20 and time.time() < deadline:
            hooke_jees(best_x, initial_step=0.1, min_step=1e-8)
        
        # Phase 3: If budget remains, try a few more random perturbations
        if eval_count < budget - 10 and time.time() < deadline:
            for _ in range(5):
                if eval_count >= budget - 5 or time.time() > deadline:
                    break
                x_perturb = best_x + rng.normal(0, 0.3, d)
                x_perturb = np.clip(x_perturb, -5, 5)
                hooke_jees(x_perturb, initial_step=0.2, min_step=1e-8, max_iter=500)

        return best_x, best_f
