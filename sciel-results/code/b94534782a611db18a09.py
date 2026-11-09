"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol initialization + Powell adaptive coordinate descent for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Sobol sequence for diverse initial points
        def sobol_points(n, dim, seed=0):
            """Generate n quasi-random points in [0,1)^dim using a simple LCG-based Sobol-like scheme."""
            points = []
            # Use a simple deterministic quasi-random sequence
            # For robustness, use Halton-like or just structured grid + random
            rng = np.random.default_rng(seed)
            # Mix: some structured + some random
            n_structured = max(1, n // 4)
            n_random = n - n_structured
            
            # Structured: Latin hypercube-like
            if n_structured > 0:
                # Create a grid
                per_axis = int(np.ceil(n_structured ** (1.0 / dim)))
                grid = np.meshgrid(*[np.linspace(0, 1, per_axis, endpoint=False) for _ in range(dim)])
                grid = np.stack([g.ravel() for g in grid], axis=1)
                # Shuffle and take first n_structured
                idx = rng.permutation(len(grid))[:n_structured]
                structured = grid[idx]
            else:
                structured = np.empty((0, dim))
            
            random_pts = rng.uniform(0, 1, size=(n_random, dim))
            points = np.vstack([structured, random_pts])
            return points

        # Generate initial candidates
        n_init = min(20, budget)
        init_unit = sobol_points(n_init, dim, seed=42)
        init_pts = -5.0 + 10.0 * init_unit  # Map to [-5, 5]^dim

        # Evaluate initial points
        best_x = None
        best_f = np.inf
        used_calls = 0

        for i in range(n_init):
            x = init_pts[i]
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Powell's method with adaptive coordinate descent
        # Powell's method: cycle through directions, update best, add new direction
        # We'll implement a modified Powell that tracks call budget

        # Initial directions: standard basis
        directions = np.eye(dim)
        # Current best point
        x_best = best_x.copy()
        f_best = best_f

        # Powell iteration
        def line_search(func, x, direction, f_current, remaining_budget):
            """Golden section line search along direction from x, minimizing func(x + t*direction)."""
            if remaining_budget < 5:
                return x, f_current, 0

            # Initial bracket: try small steps
            # For Bent Cigar, steps should be small in steep directions
            # Use adaptive step size based on previous successes
            # Simple approach: try t in [-1, 1] with golden section
            
            # Determine initial step size
            # Use a heuristic: start with step 1.0, expand if needed
            t_low = -1.0
            t_high = 1.0
            f_low = func(np.clip(x + t_low * direction, -5, 5))
            f_high = func(np.clip(x + t_high * direction, -5, 5))
            calls_used = 2
            
            # Expand bracket if needed
            step = 1.0
            while calls_used < min(8, remaining_budget):
                if f_low <= f_high:
                    # Minimum might be below t_low
                    t_new = t_low - step
                    f_new = func(np.clip(x + t_new * direction, -5, 5))
                    calls_used += 1
                    if f_new > f_low:
                        t_low, f_low = t_new, f_new
                        break
                    t_low, f_low = t_new, f_new
                    step *= 2.0
                else:
                    # Minimum might be above t_high
                    t_new = t_high + step
                    f_new = func(np.clip(x + t_new * direction, -5, 5))
                    calls_used += 1
                    if f_new > f_high:
                        t_high, f_high = t_new, f_new
                        break
                    t_high, f_high = t_new, f_new
                    step *= 2.0
            
            if calls_used >= remaining_budget:
                # Return best of bracket
                if f_low < f_high:
                    return np.clip(x + t_low * direction, -5, 5), f_low, calls_used
                else:
                    return np.clip(x + t_high * direction, -5, 5), f_high, calls_used

            # Golden section search
            gr = (np.sqrt(5) - 1) / 2  # ~0.618
            t1 = t_high - gr * (t_high - t_low)
            t2 = t_low + gr * (t_high - t_low)
            f1 = func(np.clip(x + t1 * direction, -5, 5))
            f2 = func(np.clip(x + t2 * direction, -5, 5))
            calls_used += 2

            max_iters = min(30, remaining_budget - calls_used)
            for _ in range(max_iters):
                if calls_used >= remaining_budget:
                    break
                if f1 < f2:
                    t_high = t2
                    t2 = t1
                    f2 = f1
                    t1 = t_high - gr * (t_high - t_low)
                    f1 = func(np.clip(x + t1 * direction, -5, 5))
                    calls_used += 1
                else:
                    t_low = t1
                    t1 = t2
                    f1 = f2
                    t2 = t_low + gr * (t_high - t_low)
                    f2 = func(np.clip(x + t2 * direction, -5, 5))
                    calls_used += 1

            # Best point
            if f1 < f2:
                t_best = t1
                f_best_ls = f1
            else:
                t_best = t2
                f_best_ls = f2

            x_new = np.clip(x + t_best * direction, -5, 5)
            return x_new, f_best_ls, calls_used

        # Main Powell loop
        # We'll do coordinate descent cycles
        # Track directions and their last improvement
        dir_improvements = np.zeros(dim)
        
        while used_calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Powell's method: for each direction, do line search
            # Starting from current best, go through each direction
            x_start = x_best.copy()
            f_start = f_best
            new_dir = x_start.copy()
            
            for i in range(dim):
                if used_calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                direction = directions[i]
                x_new, f_new, calls = line_search(func, x_best, direction, f_best, budget - used_calls)
                used_calls += calls
                
                if f_new < f_best:
                    # Update best
                    new_dir = x_new - x_start  # Track displacement for new direction
                    x_best = x_new
                    f_best = f_new
                    dir_improvements[i] = f_start - f_new  # How much this direction improved
                else:
                    dir_improvements[i] = 0.0

            # After all directions, check if we made progress
            if f_best < f_start:
                # Add new direction: difference between start and end of cycle
                # Replace the worst-performing direction
                new_direction = x_best - x_start
                norm_new = np.linalg.norm(new_direction)
                if norm_new > 1e-10:
                    new_direction /= norm_new
                    # Find direction with least improvement
                    worst_idx = np.argmin(dir_improvements)
                    directions[worst_idx] = new_direction
                dir_improvements = np.zeros(dim)
            else:
                # No progress: check convergence
                # If we've had several cycles without progress, stop
                break

            # Also check if step sizes are getting very small
            # (implicit convergence check)
        
        # Final polish: a few more line searches on the best directions
        # Sort directions by recent performance and do a few more passes
        if used_calls < budget and time.time() < deadline:
            # Do 2 more passes with the current directions
            for _ in range(2):
                if used_calls >= budget or time.time() > deadline:
                    break
                for i in range(dim):
                    if used_calls >= budget or time.time() > deadline:
                        break
                    direction = directions[i]
                    x_new, f_new, calls = line_search(func, x_best, direction, f_best, budget - used_calls)
                    used_calls += calls
                    if f_new < f_best:
                        x_best = x_new
                        f_best = f_new

        return x_best, f_best
# EVOLVE-BLOCK-END
