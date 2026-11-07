"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: global quadratic surrogate exploration + Hooke-Jeeves refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            call_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Global exploration with quadratic surrogate guidance ---
        # We want to find a good basin quickly. Use a sliding-window quadratic
        # model to propose candidate points, falling back to random if needed.
        
        history_X = []
        history_f = []
        max_hist = 30
        min_points = dim * (dim + 1) // 2 + 1  # 16 for dim=5
        
        # Initial random samples to populate history
        n_init = 15
        for _ in range(n_init):
            if call_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            history_X.append(x.copy())
            history_f.append(fx)

        if best_x is None:
            x0 = np.zeros(dim)
            fx0 = evaluate(x0)
            history_X.append(x0.copy())
            history_f.append(fx0)
            best_x = x0.copy()
            best_f = fx0

        # Surrogate-guided exploration loop
        # We run this for a portion of the budget to find a good starting point
        # for the local search.
        phase1_budget = min(150, budget // 3)
        while call_count < phase1_budget and time.time() < deadline:
            used_surrogate = False
            
            if len(history_X) >= min_points:
                # Build design matrix for quadratic fit
                # f(x) ≈ c + b^T x + x^T A x
                # Terms: 1, x_i, x_i*x_j (i<=j)
                n_pts = len(history_X)
                X_arr = np.array(history_X)
                F_arr = np.array(history_f)
                
                # Design matrix
                Q = np.ones((n_pts, 1 + dim + dim*(dim+1)//2))
                Q[:, 1:1+dim] = X_arr
                idx = 1 + dim
                for i in range(dim):
                    for j in range(i, dim):
                        Q[:, idx] = X_arr[:, i] * X_arr[:, j]
                        idx += 1
                
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(Q, F_arr, rcond=None)
                    
                    c = coeffs[0]
                    b = coeffs[1:1+dim]
                    A = np.zeros((dim, dim))
                    idx = 1 + dim
                    for i in range(dim):
                        for j in range(i, dim):
                            val = coeffs[idx]
                            if i == j:
                                A[i, i] = val
                            else:
                                A[i, j] = val / 2.0
                                A[j, i] = val / 2.0
                            idx += 1
                    
                    # Find minimizer of the quadratic model
                    # grad = b + 2*A*x = 0 => x* = -0.5 * A^{-1} b
                    A_reg = A + 1e-6 * np.eye(dim)
                    x_star = -0.5 * np.linalg.solve(A_reg, b)
                    x_star = np.clip(x_star, -5, 5)
                    
                    # Check if x_star is promising (inside bounds, not too far from current best?)
                    # Just evaluate it if it's distinct from existing points
                    # To avoid duplicates, we can check distance to best_x
                    if np.linalg.norm(x_star - best_x) > 1e-3:
                        fx_star = evaluate(x_star)
                        history_X.append(x_star.copy())
                        history_f.append(fx_star)
                        used_surrogate = True
                        
                        # If the surrogate prediction was good, keep going
                        # If not, we might want to explore more randomly
                        # Simple heuristic: if fx_star < best_f, we're finding the valley
                except np.linalg.LinAlgError:
                    pass

            if not used_surrogate:
                # Random exploration
                x = rng.uniform(-5, 5, dim)
                fx = evaluate(x)
                history_X.append(x.copy())
                history_f.append(fx)
            
            # Update history window
            if len(history_X) > max_hist:
                history_X.pop(0)
                history_f.pop(0)

        # --- Phase 2: Hooke-Jeeves local refinement ---
        # Start from the best point found in Phase 1
        current_x = best_x.copy()
        current_f = best_f
        
        steps = np.ones(dim)
        min_step = 1e-6
        max_step = 5.0
        step_multiplier = 2.0
        step_shrink = 0.5

        # Track coordinate success for adaptive step sizing
        coord_success = np.zeros(dim, dtype=bool)

        while call_count < budget and time.time() < deadline:
            improved = False

            for i in range(dim):
                if call_count >= budget or time.time() >= deadline:
                    break

                # Try positive direction
                trial_x = current_x.copy()
                trial_x[i] += steps[i]
                trial_x = np.clip(trial_x, -5, 5)
                f_trial = evaluate(trial_x)

                if f_trial < current_f:
                    current_x = trial_x.copy()
                    current_f = f_trial
                    improved = True
                    coord_success[i] = True
                    # Line search
                    while call_count < budget and time.time() < deadline:
                        next_x = current_x.copy()
                        next_x[i] += steps[i]
                        next_x = np.clip(next_x, -5, 5)
                        f_next = evaluate(next_x)
                        if f_next < current_f:
                            current_x = next_x.copy()
                            current_f = f_next
                        else:
                            break
                    steps[i] = min(steps[i] * step_multiplier, max_step)
                else:
                    # Try negative direction
                    trial_x = current_x.copy()
                    trial_x[i] -= steps[i]
                    trial_x = np.clip(trial_x, -5, 5)
                    f_trial = evaluate(trial_x)

                    if f_trial < current_f:
                        current_x = trial_x.copy()
                        current_f = f_trial
                        improved = True
                        coord_success[i] = True
                        # Line search
                        while call_count < budget and time.time() < deadline:
                            next_x = current_x.copy()
                            next_x[i] -= steps[i]
                            next_x = np.clip(next_x, -5, 5)
                            f_next = evaluate(next_x)
                            if f_next < current_f:
                                current_x = next_x.copy()
                                current_f = f_next
                            else:
                                break
                        steps[i] = min(steps[i] * step_multiplier, max_step)
                    else:
                        coord_success[i] = False

            if not improved:
                steps *= step_shrink
                if np.all(steps < min_step):
                    # Random restart with small noise
                    noise = rng.standard_normal(dim) * 0.5
                    new_x = best_x + noise
                    new_x = np.clip(new_x, -5, 5)
                    f_new = evaluate(new_x)
                    if f_new < current_f:
                        current_x = new_x.copy()
                        current_f = f_new
                    steps = np.ones(dim) * 0.5

            # Periodic global exploration (re-inject surrogate guidance occasionally?)
            # For now, keep it simple: just a random jump if stuck
            if call_count % 50 == 0 and call_count > 0:
                if rng.random() < 0.1:
                    jump = rng.uniform(-1, 1, dim)
                    new_x = best_x + jump
                    new_x = np.clip(new_x, -5, 5)
                    f_new = evaluate(new_x)
                    if f_new < current_f:
                        current_x = new_x.copy()
                        current_f = f_new
                    steps = np.ones(dim) * 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
