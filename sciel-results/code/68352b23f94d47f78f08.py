"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Two-phase hybrid: radial sweep for discus axis, then aggressive 1+1-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Radial sweep to find the discus axis
        # The discus function has extreme anisotropy: one narrow axis and wide perpendicular directions
        # We sweep along deterministic directions to find where the function drops rapidly
        
        # First, evaluate origin
        eval_point(np.zeros(dim))
        
        # Sweep along coordinate axes
        for i in range(dim):
            for r in [0.5, 1.0, 2.0, 3.0, 4.0]:
                if calls >= budget:
                    break
                x = np.zeros(dim)
                x[i] = r
                eval_point(x)
                x[i] = -r
                if calls >= budget:
                    break
                eval_point(x)
        
        if calls >= budget:
            return best_x, best_f

        # Find the best direction from the sweep
        # Try more directions based on where we found low values
        # Use a grid of angles in 2D planes to find the needle axis
        found_direction = None
        min_sweep_f = np.inf
        
        # Check if any axis gave good results
        for i in range(dim):
            if best_f < min_sweep_f:
                min_sweep_f = best_f
                found_direction = i
        
        # If we haven't found a good direction, try diagonal directions
        if calls < budget:
            # Try random directions to find the axis
            for _ in range(min(50, budget - calls)):
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                for r in [1.0, 2.0, 3.0]:
                    if calls >= budget:
                        break
                    x = direction * r
                    eval_point(x)
                    x = -direction * r
                    if calls >= budget:
                        break
                    eval_point(x)
        
        if calls >= budget:
            return best_x, best_f

        # Phase 2: Aggressive 1+1-ES from the best point found
        # Use a very high mutation rate to quickly converge along the narrow axis
        
        x = best_x.copy()
        f = best_f
        step_size = 0.5  # Start with moderate step
        success_count = 0
        total_steps = 0
        sigma = 0.1  # Mutation strength
        
        # Adaptive step size: increase on success, decrease on failure
        while calls < budget:
            # Generate mutation
            mutation = rng.standard_normal(dim)
            new_x = x + step_size * mutation
            new_x = np.clip(new_x, -5, 5)
            
            new_f = eval_point(new_x)
            total_steps += 1
            
            if new_f < f:
                # Success: move to new point, increase step size
                x = new_x
                f = new_f
                step_size = min(step_size * 1.1, 5.0)
                success_count += 1
            else:
                # Failure: decrease step size
                step_size = max(step_size * 0.9, 0.001)
            
            # Adapt mutation strength based on success rate
            if total_steps > 10:
                success_rate = success_count / total_steps
                if success_rate > 0.35:
                    sigma *= 1.05
                elif success_rate < 0.15:
                    sigma *= 0.95
                sigma = np.clip(sigma, 0.01, 1.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
