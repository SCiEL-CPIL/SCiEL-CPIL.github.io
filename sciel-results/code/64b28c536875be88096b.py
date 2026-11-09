"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using Adaptive Coordinate Descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Coordinate Descent with random perturbations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            try:
                val = func(x)
                if not np.isfinite(val):
                    return 1e10
                return float(val)
            except Exception:
                return 1e10

        def eval_pt(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = safe_func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # 1. Initial Phase: Sparse random sampling to find a promising basin
        # Lunacek has two basins. A few random points help identify the better one.
        n_init = 10
        for _ in range(n_init):
            if call_count >= budget or time.time() > deadline:
                break
            x_init = rng.uniform(-5, 5, dim)
            eval_pt(x_init)

        # 2. Main Phase: Adaptive Coordinate Descent
        # We maintain a current point `x` and its function value `fx`.
        # We try to improve by moving along each coordinate axis.
        
        x = best_x.copy()
        fx = best_f
        step_sizes = np.ones(dim) * 1.0  # Initial step size
        global_step_decay = 0.95
        min_step = 1e-6
        max_step = 5.0
        stagnation_count = 0
        perturb_every = 30  # Perturb if no improvement for this many steps
        no_improve_count = 0

        while call_count < budget and time.time() < deadline:
            improved_in_cycle = False
            
            for i in range(dim):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Try positive step
                x_new = x.copy()
                x_new[i] += step_sizes[i]
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = eval_pt(x_new)
                
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    improved_in_cycle = True
                    no_improve_count = 0
                    # Increase step size if improvement was significant? 
                    # For now, keep step size or slightly increase if we are far from optimum.
                    # Simple strategy: keep step size.
                else:
                    # Try negative step
                    x_new = x.copy()
                    x_new[i] -= step_sizes[i]
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = eval_pt(x_new)
                    
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        improved_in_cycle = True
                        no_improve_count = 0
                    else:
                        # No improvement in this coordinate, shrink step size
                        step_sizes[i] *= 0.8
                        if step_sizes[i] < min_step:
                            step_sizes[i] = min_step

            # If no improvement in the entire cycle, increment stagnation
            if not improved_in_cycle:
                no_improve_count += 1
                
                # If stagnating, perturb the position to escape local minimum
                if no_improve_count >= perturb_every:
                    # Random jump
                    jump_scale = 1.0
                    x_perturbed = x + rng.normal(0, jump_scale, dim)
                    x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                    f_perturbed = eval_pt(x_perturbed)
                    
                    if f_perturbed < fx:
                        x = x_perturbed
                        fx = f_perturbed
                        no_improve_count = 0
                        # Reset step sizes to moderate values after a successful jump
                        step_sizes = np.ones(dim) * 1.0
                    else:
                        # Even perturbation didn't help, reduce global step size
                        step_sizes *= global_step_decay
                        no_improve_count = 0 # Reset counter to try again

        return best_x, best_f
# EVOLVE-BLOCK-END
