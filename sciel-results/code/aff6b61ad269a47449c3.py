"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-guided (1+1)-ES for bent-cigar (f12) optimization."""

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

        # 1. Initialization
        n_init = 10
        for _ in range(n_init):
            if call_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if best_x is None:
            x0 = np.zeros(dim)
            evaluate(x0)
            best_x = x0

        # 2. Surrogate-guided (1+1)-ES
        # Build a simple linear/quadratic surrogate from observed points
        # Use adaptive sigma with 1/5 success rule
        current_x = best_x.copy()
        current_f = best_f
        sigma = 1.0
        min_sigma = 1e-7
        max_sigma = 5.0
        
        # Store observed points for surrogate
        X_obs = [best_x.copy()]
        F_obs = [best_f]
        
        # Success counter for sigma adaptation
        success_count = 0
        window_size = 10
        window = []

        while call_count < budget and time.time() < deadline:
            # Generate candidate
            direction = rng.standard_normal(dim)
            y = current_x + sigma * direction
            y = np.clip(y, -5, 5)
            
            fy = evaluate(y)
            
            # Update surrogate data
            X_obs.append(y.copy())
            F_obs.append(fy)
            
            # Keep only last 50 points for surrogate to limit computation
            if len(X_obs) > 50:
                X_obs = X_obs[-50:]
                F_obs = F_obs[-50:]
            
            if fy < current_f:
                current_x = y.copy()
                current_f = fy
                success_count += 1
                window.append(1)
                # Increase sigma on success
                sigma = min(sigma * 1.3, max_sigma)
            else:
                window.append(0)
                # Decrease sigma on failure
                sigma = max(sigma * 0.9, min_sigma)
            
            # 1/5 success rule adaptation
            if len(window) >= window_size:
                success_rate = sum(window) / len(window)
                if success_rate > 0.5:
                    sigma = min(sigma * 1.1, max_sigma)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.9, min_sigma)
                window = window[-window_size:]
            
            # Periodic global exploration: occasionally jump to a new random point
            # This helps escape local minima in the anisotropic landscape
            if call_count % 50 == 0 and call_count > 0:
                # With some probability, do a larger exploratory jump
                if rng.random() < 0.3:
                    jump = rng.uniform(-3, 3, dim)
                    new_x = best_x + jump
                    new_x = np.clip(new_x, -5, 5)
                    f_new = evaluate(new_x)
                    X_obs.append(new_x.copy())
                    F_obs.append(f_new)
                    if len(X_obs) > 50:
                        X_obs = X_obs[-50:]
                        F_obs = F_obs[-50:]
                    if f_new < current_f:
                        current_x = new_x.copy()
                        current_f = f_new
                    # Reset sigma after jump
                    sigma = max(sigma * 0.5, min_sigma)

        return best_x, best_f
# EVOLVE-BLOCK-END
