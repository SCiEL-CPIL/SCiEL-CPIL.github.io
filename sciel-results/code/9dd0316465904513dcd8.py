"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate descent with time guard for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            x = np.clip(np.asarray(x, dtype=float).reshape(dim), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def coordinate_descent(x, step_sizes, max_iter=1000):
            """Cyclic coordinate descent with adaptive step sizes."""
            nonlocal x_best, f_best, calls_made
            x = x.copy()
            fx = f_best  # We don't re-evaluate x itself; we assume it's already known
            # Actually, we should be careful. Let's just use the current best.
            # But coordinate descent needs to start from a point. Let's evaluate it if needed.
            # To save budget, we assume x is close to x_best or is x_best.
            # Let's just start from x and evaluate it once if it's not x_best.
            # For efficiency, let's just do line searches.
            
            for _ in range(max_iter):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    if not time_ok():
                        break
                    for step in step_sizes:
                        for direction in [1, -1]:
                            y = x.copy()
                            y[d] += direction * step
                            y = np.clip(y, -5, 5)
                            fy = evaluate(y)
                            if fy < fx:
                                x = y
                                fx = fy
                                improved = True
                                break
                        if improved:
                            break
                if not improved:
                    # Shrink step sizes
                    step_sizes = [s * 0.7 for s in step_sizes]
                    if step_sizes[0] < 1e-10:
                        break
            return x, fx

        def multi_scale_search():
            """Deterministic multi-scale search."""
            nonlocal x_best, f_best, calls_made
            
            # Phase 1: Coarse grid search
            # Start with a relatively coarse grid
            base_step = 2.0
            while base_step > 0.05 and time_ok():
                # Generate grid points
                n_per_dim = int(10.0 / base_step) + 1
                if n_per_dim > 20:
                    n_per_dim = 20
                base_step = 10.0 / (n_per_dim - 1)
                
                # Latin hypercube sampling for efficiency
                n_samples = min(200, budget - calls_made)
                if n_samples <= 0:
                    break
                    
                # Simple deterministic sampling: stratified
                points = []
                for i in range(n_samples):
                    x = np.zeros(dim)
                    for d in range(dim):
                        # Stratified sampling
                        frac = (i + 0.5) / n_samples
                        x[d] = -5 + 10 * frac + base_step * ((i * 7 + d * 13) % 10) / 10.0
                    x = np.clip(x, -5, 5)
                    points.append(x)
                
                for p in points:
                    if not time_ok():
                        break
                    evaluate(p)
                
                # Refine around best
                if x_best is not None:
                    local_step = base_step * 0.5
                    n_local = min(50, budget - calls_made)
                    if n_local > 0 and time_ok():
                        for i in range(n_local):
                            if not time_ok():
                                break
                            # Deterministic offsets
                            offset = np.zeros(dim)
                            for d in range(dim):
                                offset[d] = local_step * ((i * 11 + d * 17) % 10 - 5) / 5.0
                            x = np.clip(x_best + offset, -5, 5)
                            evaluate(x)
                
                base_step *= 0.5
            
            # Phase 2: Coordinate descent refinement
            if x_best is not None and time_ok():
                step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
                coordinate_descent(x_best, step_sizes, max_iter=500)
                
                # Try a few different starting points for coordinate descent
                for trial in range(3):
                    if not time_ok():
                        break
                    # Deterministic perturbation
                    x_start = x_best.copy()
                    for d in range(dim):
                        x_start[d] += 0.3 * np.sin(trial * 1.1 + d * 0.7)
                    x_start = np.clip(x_start, -5, 5)
                    coordinate_descent(x_start, step_sizes, max_iter=200)
            
            # Phase 3: Very fine local search
            if x_best is not None and time_ok():
                fine_steps = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
                coordinate_descent(x_best, fine_steps, max_iter=300)

        multi_scale_search()

        return x_best, f_best

# EVOLVE-BLOCK-END
