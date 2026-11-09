"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Fidelity SQP with Piecewise-Linear Surrogate for BBOB f6.
    
    Structural approach:
    1. Coarse structured sampling to identify the attractive sector
    2. Build a piecewise-linear surrogate from observations
    3. Fast local line searches within the identified sector
    4. Coarse-to-fine grid scan to refine global position
    5. Adaptive local search with 1/5 success rule
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5 second safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Store observations for model building
        X_obs = []
        f_obs = []

        def eval_and_update(x):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            f = func(x)
            X_obs.append(x.copy())
            f_obs.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Structured sampling to identify the attractive sector
        center = np.zeros(dim)
        eval_and_update(center)

        # Sample along each axis in both directions with multiple radii
        for i in range(dim):
            for delta in [0.5, 1.0, 2.0, 3.0, -0.5, -1.0, -2.0, -3.0]:
                if calls_made >= budget:
                    break
                x = center.copy()
                x[i] = delta
                x = np.clip(x, -5, 5)
                eval_and_update(x)

        # Sample along diagonal directions
        for _ in range(15):
            if calls_made >= budget:
                break
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction) + 1e-15
            for radius in [0.5, 1.0, 2.0, 3.0, 4.0]:
                if calls_made >= budget:
                    break
                x = np.clip(center + radius * direction, -5, 5)
                eval_and_update(x)

        # Add some random points for broader coverage
        for _ in range(20):
            if calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            eval_and_update(x)

        # Phase 2: Identify the sector direction from observations
        if len(X_obs) >= 10:
            X_arr = np.array(X_obs)
            f_arr = np.array(f_obs)

            # Weight by how much better each point is than the worst
            f_min = f_arr.min()
            f_max = f_arr.max()
            if f_max > f_min:
                weights = (f_max - f_arr) / (f_max - f_min + 1e-10)
            else:
                weights = np.ones(len(f_arr)) / len(f_arr)

            # Compute weighted direction from center
            directions = X_arr - center
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            unit_dirs = directions / norms

            # Weighted sum of unit directions
            sector_dir = np.sum(weights[:, None] * unit_dirs, axis=0)
            sector_norm = np.linalg.norm(sector_dir)
            if sector_norm > 1e-6:
                sector_dir /= sector_norm
            else:
                sector_dir = rng.standard_normal(dim)
                sector_dir /= np.linalg.norm(sector_dir)
        else:
            sector_dir = rng.standard_normal(dim)
            sector_dir /= np.linalg.norm(sector_dir)

        # Phase 3: Build piecewise-linear surrogate and perform fast local searches
        # Fit a local linear model around the best point
        x0 = best_x.copy()

        # Find the k nearest neighbors to build a local linear model
        k = min(20, len(X_obs))
        if k >= dim + 1:
            X_arr = np.array(X_obs)
            f_arr = np.array(f_obs)

            # Compute distances to x0
            dists = np.linalg.norm(X_arr - x0, axis=1)
            idx = np.argsort(dists)[:k]
            X_near = X_arr[idx]
            f_near = f_arr[idx]

            # Fit linear model: f(x) ≈ a + b^T (x - x0)
            dX = X_near - x0
            df = f_near - f_near[0]

            # Solve: dX @ b = df (with intercept)
            A = np.hstack([dX, np.ones((k, 1))])
            try:
                params, residuals, rank, sv = np.linalg.lstsq(A, df, rcond=None)
                b_grad = params[:dim]
                a_intercept = params[dim]

                # Use the gradient direction for line searches
                grad_norm = np.linalg.norm(b_grad)
                if grad_norm > 1e-6:
                    step_dir = -b_grad / grad_norm

                    # Try several step sizes along the gradient direction
                    for step_size in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
                        if calls_made >= budget or time.time() >= deadline:
                            break
                        x_trial = np.clip(x0 + step_size * step_dir, -5, 5)
                        f_trial = eval_and_update(x_trial)
                        if f_trial < best_f:
                            x0 = best_x.copy()
                            # Continue with smaller steps from the new best
                            break
            except:
                pass

        # Phase 4: Coarse-to-fine grid scan around the best point
        # Start with a coarse grid, then refine
        base_point = best_x.copy()

        # Coarse grid: step size 1.0
        if calls_made < budget - 50:
            step_coarse = 1.0
            offsets = []
            for i in range(dim):
                for delta in [-step_coarse, 0, step_coarse]:
                    if delta == 0:
                        continue
                    offset = np.zeros(dim)
                    offset[i] = delta
                    offsets.append(offset)

            # Also add some diagonal offsets
            for _ in range(5):
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction) + 1e-15
                offsets.append(step_coarse * direction)

            for offset in offsets:
                if calls_made >= budget or time.time() >= deadline:
                    break
                x_trial = np.clip(base_point + offset, -5, 5)
                eval_and_update(x_trial)

        # Fine grid: step size 0.3
        if calls_made < budget - 30:
            step_fine = 0.3
            offsets = []
            for i in range(dim):
                for delta in [-step_fine, step_fine]:
                    offset = np.zeros(dim)
                    offset[i] = delta
                    offsets.append(offset)

            for offset in offsets:
                if calls_made >= budget or time.time() >= deadline:
                    break
                x_trial = np.clip(base_point + offset, -5, 5)
                eval_and_update(x_trial)

        # Phase 5: Final adaptive refinement with 1/5 success rule
        sigma = 0.5
        successes = 0
        trials = 0

        while calls_made < budget and time.time() < deadline:
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction) + 1e-15

            x_trial = np.clip(best_x + sigma * direction, -5, 5)
            f_trial = eval_and_update(x_trial)

            if f_trial < best_f:
                successes += 1
                sigma *= 1.5  # 1/5 rule: increase on success
            else:
                sigma *= 0.87  # 1/5 rule: decrease on failure

            trials += 1
            if trials >= 5:
                # Check success rate
                rate = successes / trials
                if rate > 0.5:
                    sigma *= 1.2
                elif rate < 0.2:
                    sigma *= 0.7
                successes = 0
                trials = 0

            # If sigma is too small, try a restart
            if sigma < 0.01:
                # Try a larger jump
                if calls_made < budget - 1:
                    big_jump = rng.standard_normal(dim) * rng.uniform(1.0, 3.0)
                    x_big = np.clip(best_x + big_jump, -5, 5)
                    eval_and_update(x_big)
                sigma = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
