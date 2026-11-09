"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Piecewise-linear surrogate model optimizer for BBOB f6 (Attractive Sector).
    
    Structural approach:
    1. Coarse sampling to identify the attractive sector direction
    2. Build a piecewise-linear surrogate from observations
    3. Line search within the identified sector using the surrogate
    4. Refine with adaptive local search
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

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
        # The f6 function has a sector-shaped region of attraction
        # We sample along principal axes and diagonals to detect the sector

        center = np.zeros(dim)
        eval_and_update(center)

        # Sample along each axis in both directions
        for i in range(dim):
            for delta in [1.0, -1.0, 2.0, -2.0]:
                if calls_made >= budget:
                    break
                x = center.copy()
                x[i] = delta
                x = np.clip(x, -5, 5)
                eval_and_update(x)

        # Sample along diagonal directions
        for _ in range(10):
            if calls_made >= budget:
                break
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction) + 1e-15
            for radius in [1.0, 2.0, 3.0]:
                x = np.clip(center + radius * direction, -5, 5)
                eval_and_update(x)

        # Phase 2: Identify the sector direction from observations
        # Points with lower f values indicate the attractive sector
        # Compute a weighted average of directions from center to low-f points
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

        # Phase 3: Line search along the sector direction
        # Sample points along the sector direction to find the optimum
        # Use golden section or binary search on the 1D function along this line
        
        # First, bracket the minimum along the sector direction
        t_low, t_high = -5.0, 5.0
        best_t = 0.0
        best_f_t = best_f
        
        # Sample along the line to bracket
        n_line_samples = min(20, budget - calls_made)
        if n_line_samples > 0:
            t_vals = np.linspace(-4, 4, n_line_samples)
            for t in t_vals:
                if calls_made >= budget:
                    break
                x = np.clip(center + t * sector_dir, -5, 5)
                f = eval_and_update(x)
                if f < best_f_t:
                    best_f_t = f
                    best_t = t

            # Refine with golden section search
            if calls_made < budget - 10:
                # Golden section search on [t_low, t_high]
                gr = (np.sqrt(5) - 1) / 2  # ~0.618
                a, b = -4.0, 4.0
                
                # Evaluate at golden points
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                x_c = np.clip(center + c * sector_dir, -5, 5)
                x_d = np.clip(center + d * sector_dir, -5, 5)
                f_c = eval_and_update(x_c)
                f_d = eval_and_update(x_d)
                
                # Iterative refinement
                for _ in range(15):
                    if calls_made >= budget - 5 or time.time() >= deadline:
                        break
                    if f_c < f_d:
                        b = d
                        d = c
                        f_d = f_c
                        c = b - gr * (b - a)
                        x_c = np.clip(center + c * sector_dir, -5, 5)
                        f_c = eval_and_update(x_c)
                    else:
                        a = c
                        c = d
                        f_c = f_d
                        d = a + gr * (b - a)
                        x_d = np.clip(center + d * sector_dir, -5, 5)
                        f_d = eval_and_update(x_d)
                
                # Best point from golden section
                if f_c < f_d:
                    best_t = c
                else:
                    best_t = d

        # Phase 4: Build piecewise-linear surrogate and refine
        # Use the observations to build a local linear model around the best point
        x0 = best_x.copy()
        
        # Find the k nearest neighbors to build a local linear model
        k = min(15, len(X_obs))
        if k >= dim + 1:
            X_arr = np.array(X_obs)
            f_arr = np.array(f_obs)
            
            # Compute distances to x0
            dists = np.linalg.norm(X_arr - x0, axis=1)
            idx = np.argsort(dists)[:k]
            X_near = X_arr[idx]
            f_near = f_arr[idx]
            
            # Fit linear model: f(x) ≈ a + b^T (x - x0)
            # Using least squares
            dX = X_near - x0
            df = f_near - f_near[0]  # Relative to first point
            
            # Solve: dX @ b = df (with intercept)
            A = np.hstack([dX, np.ones((k, 1))])
            try:
                params, residuals, rank, sv = np.linalg.lstsq(A, df, rcond=None)
                b_grad = params[:dim]
                a_intercept = params[dim]
                
                # The minimum of the linear model is at infinity unless we constrain
                # Instead, use the gradient direction for a line search
                grad_norm = np.linalg.norm(b_grad)
                if grad_norm > 1e-6:
                    # Move in the negative gradient direction
                    step_dir = -b_grad / grad_norm
                    # Try several step sizes
                    for step_size in [0.5, 1.0, 2.0, 3.0]:
                        if calls_made >= budget or time.time() >= deadline:
                            break
                        x_trial = np.clip(x0 + step_size * step_dir, -5, 5)
                        f_trial = eval_and_update(x_trial)
                        if f_trial < best_f:
                            # Success, continue with smaller steps
                            break
            except:
                pass

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
