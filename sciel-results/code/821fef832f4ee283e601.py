"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse grid search followed by Powell's method."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        calls = [0]

        def call_func(x):
            if calls[0] >= budget:
                return float('inf')
            calls[0] += 1
            return float(func(x))

        def time_ok():
            return time.time() < deadline

        # Phase 1: Coarse grid search to find good starting points
        # Use a moderate resolution grid to cover [-5, 5]^5
        # With 500 calls total, we need to be efficient
        # Let's use a grid of about 3^5 = 243 points for the coarse phase
        # Then use remaining budget for local optimization
        
        n_grid = 3  # points per dimension for coarse search
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    for l in range(n_grid):
                        for m in range(n_grid):
                            x = np.array([
                                -5.0 + 10.0 * i / (n_grid - 1),
                                -5.0 + 10.0 * j / (n_grid - 1),
                                -5.0 + 10.0 * k / (n_grid - 1),
                                -5.0 + 10.0 * l / (n_grid - 1),
                                -5.0 + 10.0 * m / (n_grid - 1)
                            ])
                            grid_points.append(x)

        # Evaluate all grid points
        best_x = None
        best_f = float('inf')
        for x in grid_points:
            if not time_ok():
                break
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = call_func(best_x)

        # Phase 2: Powell's method local optimization
        # Implement Powell's method manually to track function calls
        def powell_local(x0, f0, max_iter=50):
            """Powell's method with call tracking."""
            n = len(x0)
            x = x0.copy()
            f = f0
            # Initial directions: standard basis
            directions = [np.eye(n)[i] for i in range(n)]
            
            for it in range(max_iter):
                if not time_ok() or calls[0] >= budget:
                    break
                
                old_x = x.copy()
                old_f = f
                
                # Move along each direction
                for i in range(n):
                    # Find optimal step along direction i
                    d = directions[i]
                    # Try a few step sizes
                    best_t = 0.0
                    best_f_dir = f
                    
                    # Initial step size
                    t = 0.5
                    # Try positive and negative directions
                    for _ in range(10):
                        if calls[0] >= budget:
                            break
                        x_trial = x + t * d
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = call_func(x_trial)
                        if f_trial < best_f_dir:
                            best_f_dir = f_trial
                            best_t = t
                        t *= 0.5
                    
                    # Also try negative direction
                    t = -0.5
                    for _ in range(10):
                        if calls[0] >= budget:
                            break
                        x_trial = x + t * d
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = call_func(x_trial)
                        if f_trial < best_f_dir:
                            best_f_dir = f_trial
                            best_t = t
                        t *= 0.5
                    
                    # Move to best point along this direction
                    x = x + best_t * d
                    x = np.clip(x, -5, 5)
                    f = best_f_dir
                    
                    # Update direction: new direction is from old_x to current x
                    new_dir = x - old_x
                    if np.linalg.norm(new_dir) > 1e-10:
                        directions[i] = new_dir / np.linalg.norm(new_dir)
                
                # Check for convergence
                if abs(f - old_f) < 1e-10:
                    break
                
                # Update old_x for next iteration
                old_x = x.copy()
                old_f = f
                
                # Add new direction: from old_x (start of this iteration) to current x
                new_dir = x - old_x
                if np.linalg.norm(new_dir) > 1e-10:
                    new_dir = new_dir / np.linalg.norm(new_dir)
                    # Replace the direction that gave the largest improvement
                    # For simplicity, just append and keep last n+1 directions
                    directions.append(new_dir)
                    if len(directions) > n + 1:
                        # Remove the oldest direction (or the one with smallest improvement)
                        # For now, just remove the first one
                        directions.pop(0)
                
                # Reorder directions to keep the most effective ones
                # Simple approach: keep all, but limit total
            
            return x, f

        # Run Powell's method from the best grid point
        if time_ok() and calls[0] < budget:
            x_opt, f_opt = powell_local(best_x, best_f, max_iter=30)
            if f_opt < best_f:
                best_x = x_opt
                best_f = f_opt

        # If we still have budget, try a few more local optimizations from slightly perturbed points
        if time_ok() and calls[0] < budget - 10:
            rng = np.random.default_rng(42)
            for _ in range(5):
                if not time_ok() or calls[0] >= budget:
                    break
                # Perturb the best point
                x_perturb = best_x + 0.5 * rng.standard_normal(dim)
                x_perturb = np.clip(x_perturb, -5, 5)
                f_perturb = call_func(x_perturb)
                
                if f_perturb < best_f:
                    x_opt, f_opt = powell_local(x_perturb, f_perturb, max_iter=20)
                    if f_opt < best_f:
                        best_x = x_opt
                        best_f = f_opt

        return best_x, best_f
# EVOLVE-BLOCK-END
