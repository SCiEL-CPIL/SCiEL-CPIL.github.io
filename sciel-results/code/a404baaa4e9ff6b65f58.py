"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Derivative-Free Broyden (DFB) for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        n = self.dim
        max_calls = self.budget
        total_calls = 0
        best_x = None
        best_f = np.inf

        def time_check():
            return time.time() < deadline

        def safe_func(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= max_calls:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            total_calls += 1
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        rng = np.random.default_rng(7)

        def dfb_local(x0, f0, budget_per_run, seed_offset=0):
            """Run DFB local optimization from x0 with f0."""
            nonlocal total_calls, best_x, best_f
            if not time_check() or total_calls >= max_calls:
                return

            rng_local = np.random.default_rng(100 + seed_offset)
            x = x0.copy()
            f = f0
            H = np.eye(n) * 1.0  # Broyden Hessian approximation
            max_iter = 30

            for it in range(max_iter):
                if total_calls >= max_calls or not time_check():
                    break

                # Compute step: p = -H * grad_approx
                # For derivative-free, use a line search approach
                # Try moving in the direction suggested by H
                # Use a simple finite-difference-like step
                step_size = 0.5 * (1.0 + it * 0.1)
                
                # Generate several candidate directions
                best_trial_f = np.inf
                best_trial_x = None
                
                # Direction 1: along -H * (x - x0) normalized (pull toward center if needed)
                # Direction 2: random directions scaled by H
                for _ in range(8):
                    if total_calls >= max_calls or not time_check():
                        break
                    
                    if _ == 0:
                        # Move toward origin (common for ellipsoidal functions)
                        direction = -x
                    else:
                        direction = rng_local.standard_normal(n)
                    
                    # Scale direction by H
                    direction = H @ direction
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        direction = rng_local.standard_normal(n)
                        norm = np.linalg.norm(direction)
                    direction = direction / norm * step_size
                    
                    x_trial = np.clip(x + direction, -5.0, 5.0)
                    f_trial = safe_func(x_trial)
                    
                    if np.isfinite(f_trial) and f_trial < best_trial_f:
                        best_trial_f = f_trial
                        best_trial_x = x_trial.copy()
                
                if best_trial_x is None or best_trial_f >= f - 1e-12:
                    # No improvement, shrink step
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
                    continue
                
                # Update H using Broyden formula
                # p = x_trial - x, f_diff = f_trial - f
                p = best_trial_x - x
                f_diff = best_trial_f - f
                
                # Simple Broyden update: H_new = H + (p - H*grad_approx) * ... 
                # For simplicity, use a rank-1 update based on the step
                # This is a simplified version; full DFB would be more complex
                if np.linalg.norm(p) > 1e-10:
                    H = H + np.outer(p, p) * 0.1  # Simple rank-1 update
                
                x = best_trial_x
                f = best_trial_f
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            return x, f

        # --- Phase 1: Coarse random search to identify promising regions ---
        n_scan = min(30, max_calls // 8)
        scan_points = [np.zeros(n)]
        for _ in range(n_scan - 1):
            scan_points.append(rng.uniform(-4, 4, n))

        scan_fitness = []
        for p in scan_points:
            if total_calls >= max_calls or not time_check():
                break
            f = safe_func(p)
            scan_fitness.append(f)

        # --- Phase 2: Multi-Start DFB from diverse starting points ---
        # Select diverse starting points from scan results
        if scan_fitness:
            sorted_scan = np.argsort(scan_fitness)
            n_starts = min(8, len(sorted_scan))
            
            # Ensure diversity: pick from top performers but spread out
            start_points = []
            for idx in sorted_scan[:n_starts]:
                start_points.append((scan_points[idx].copy(), scan_fitness[idx]))
            
            # Add some random starts for diversity
            for _ in range(4):
                if total_calls < max_calls and time_check():
                    x = rng.uniform(-4, 4, n)
                    f = safe_func(x)
                    if np.isfinite(f):
                        start_points.append((x, f))
            
            # Run DFB from each start point with a budget
            budget_per_run = max(10, (max_calls - total_calls) // max(1, len(start_points)))
            
            for i, (x0, f0) in enumerate(start_points):
                if total_calls >= max_calls or not time_check():
                    break
                
                x_opt, f_opt = dfb_local(x0, f0, budget_per_run, seed_offset=i)
                
                if x_opt is not None and f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()
        
        # --- Phase 3: Final refinement with Nelder-Mead if budget remains ---
        if best_x is not None and total_calls < max_calls and time_check():
            try:
                from scipy.optimize import minimize

                def objective(x):
                    return safe_func(x)

                # Start Nelder-Mead from current best
                res = minimize(objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 30, 'xatol': 1e-5, 'fatol': 1e-7})

                if res.success and np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)

            except Exception:
                pass

        # --- Phase 4: Final coordinate descent refinement ---
        if best_x is not None and total_calls < max_calls and time_check():
            x_curr = best_x.copy()
            f_curr = best_f
            step = 0.2
            stall_count = 0

            for _ in range(60):
                if total_calls >= max_calls or not time_check():
                    break

                improved = False
                for dim_idx in range(n):
                    for direction in [-1, 1]:
                        x_trial = x_curr.copy()
                        x_trial[dim_idx] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        if np.isfinite(f_trial) and f_trial < f_curr - 1e-12:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                            stall_count = 0
                            break
                    if improved:
                        break

                if improved:
                    step = min(step * 1.05, 1.0)
                else:
                    step *= 0.6
                    stall_count += 1
                    if step < 1e-5 or stall_count > 10:
                        break

            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr

        if best_x is None:
            best_x = np.zeros(n)
            if total_calls < max_calls:
                best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
