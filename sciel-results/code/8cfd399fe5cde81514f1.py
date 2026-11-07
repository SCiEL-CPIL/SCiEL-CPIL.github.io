import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration with stratified sampling
        # f13 minimum is in [-4, 4]^5
        n_global = 15
        for _ in range(n_global):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            safe_call(x)

        if best_x is None:
            x0 = np.zeros(dim)
            safe_call(x0)
            return best_x, best_f

        # Phase 2: Coordinate Descent with adaptive step sizes
        # This leverages the anisotropic nature of f13 (sharp ridge)
        # by adapting step sizes per dimension based on recent progress

        def coordinate_descent(x_start, f_start, max_iter=200):
            nonlocal calls, best_f, best_x

            x = x_start.copy()
            f_curr = f_start
            # Adaptive step sizes per dimension
            steps = np.ones(dim) * 0.5
            # Track recent improvements per dimension
            recent_improve = np.ones(dim) * 0.5
            stagnant = 0
            max_stagnant = 30

            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break

                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Try positive direction
                    x_trial = x.copy()
                    x_trial[d] += steps[d]
                    f_trial = safe_call(x_trial)
                    if f_trial is None:
                        break

                    if f_trial < f_curr:
                        x = x_trial
                        f_curr = f_trial
                        improved = True
                        # Adapt step size: increase if improving
                        steps[d] = min(steps[d] * 1.5, 2.0)
                        recent_improve[d] = 0.0
                    else:
                        # Try negative direction
                        x_trial = x.copy()
                        x_trial[d] -= steps[d]
                        f_trial = safe_call(x_trial)
                        if f_trial is None:
                            break

                        if f_trial < f_curr:
                            x = x_trial
                            f_curr = f_trial
                            improved = True
                            # Adapt step size: increase if improving
                            steps[d] = min(steps[d] * 1.5, 2.0)
                            recent_improve[d] = 0.0
                        else:
                            # No improvement in this dimension
                            steps[d] = max(steps[d] * 0.7, 0.01)
                            recent_improve[d] += 1.0

                    # Update global best
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x.copy()

                # Stagnation detection
                if not improved:
                    stagnant += 1
                    if stagnant >= max_stagnant:
                        # Reset step sizes and break to restart
                        break
                else:
                    stagnant = 0

                # Convergence check
                if np.all(steps < 0.02):
                    break

            return x, f_curr

        # Run coordinate descent from best point
        n_restarts = 8
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            if r == 0:
                # First restart from current best
                x_cd, f_cd = coordinate_descent(best_x.copy(), best_f)
            else:
                # Restart strategy: mix of random and perturbed points
                if rng.random() < 0.4:
                    # Random point
                    x_new = rng.uniform(-5.0, 5.0, size=dim)
                elif rng.random() < 0.7:
                    # Perturb best
                    x_new = best_x + rng.normal(0, 0.3, size=dim)
                else:
                    # Intermediate point between best and random
                    x_rand = rng.uniform(-5.0, 5.0, size=dim)
                    x_new = 0.5 * best_x + 0.5 * x_rand

                f_new = safe_call(x_new)
                if f_new is None:
                    break

                # Only run CD if this point is promising
                if f_new < best_f + 5.0:  # Within some tolerance
                    x_cd, f_cd = coordinate_descent(x_new, f_new)

        # Phase 3: Final refinement with fine coordinate descent
        if calls < budget and time.time() < deadline:
            # Use very small steps for final polishing
            x_final = best_x.copy()
            f_final = best_f
            fine_steps = np.ones(dim) * 0.05
            for _ in range(50):
                if calls >= budget or time.time() > deadline:
                    break
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Try small positive step
                    x_trial = x_final.copy()
                    x_trial[d] += fine_steps[d]
                    f_trial = safe_call(x_trial)
                    if f_trial is not None and f_trial < f_final:
                        x_final = x_trial
                        f_final = f_trial
                    else:
                        # Try small negative step
                        x_trial = x_final.copy()
                        x_trial[d] -= fine_steps[d]
                        f_trial = safe_call(x_trial)
                        if f_trial is not None and f_trial < f_final:
                            x_final = x_trial
                            f_final = f_trial
                        else:
                            # Shrink step for this dimension
                            fine_steps[d] *= 0.5

                if f_final < best_f:
                    best_f = f_final
                    best_x = x_final.copy()

                # Check convergence
                if np.all(fine_steps < 0.001):
                    break

        return best_x, best_f
