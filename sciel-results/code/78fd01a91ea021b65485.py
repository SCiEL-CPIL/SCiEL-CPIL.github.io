import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        # --- Phase 1: Initial Exploration ---
        # Use Halton sequence for space-filling initial points
        def halton_sequence(n, dim):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                f = 1.0
                i = 1
                for j in range(n):
                    f /= base
                    i += 1
                    while i % base == 0:
                        i //= base
                        f *= base
                    points[j, d] = f
            return points

        n_init = min(30, budget)
        if budget <= n_init:
            # Very small budget: just do random search
            best_x = None
            best_f = np.inf
            for _ in range(budget):
                x = np.random.uniform(-5, 5, dim)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            return best_x, best_f

        # Generate initial points using Halton
        H = halton_sequence(n_init, dim)
        X_init = H * 10.0 - 5.0  # Map to [-5, 5]^d

        # Evaluate initial points
        F_init = np.array([func(x) for x in X_init])
        evals = n_init

        best_idx = np.argmin(F_init)
        best_x = X_init[best_idx].copy()
        best_f = F_init[best_idx]

        # --- Phase 2: Multi-start Stochastic Hill Climbing with Adaptive Steps ---
        # For Schaffers F7 (ill-conditioned), axis-aligned search with adaptive
        # per-dimension steps is more efficient than full covariance updates.

        # Number of walkers for multi-start
        n_walkers = 5
        # Each walker starts from the best initial points
        walker_starts = X_init[np.argsort(F_init)[:n_walkers]]

        # Per-dimension step sizes (adaptive)
        # Initialize with moderate steps
        base_step = 0.5
        steps = np.full(dim, base_step)

        # Momentum for each walker and dimension
        momenta = np.zeros((n_walkers, dim))

        # Learning rate for step adaptation
        step_lr = 0.9

        # No-improvement counter
        no_improve_count = 0
        max_no_improve = 20

        while evals < budget:
            if time.time() > deadline:
                break

            walker_improved = False

            for w in range(n_walkers):
                if evals >= budget or time.time() > deadline:
                    break

                x = walker_starts[w].copy()
                f = func(x)  # Re-evaluate to get current value
                evals += 1

                # If this walker finds a new global best, update
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    walker_improved = True

                # Coordinate-wise line search with adaptive steps
                for d in range(dim):
                    if evals >= budget or time.time() > deadline:
                        break

                    # Try positive and negative directions
                    best_trial_f = f
                    best_trial_x = x.copy()

                    for direction in [-1, 1]:
                        # Adaptive step size with momentum
                        step_size = steps[d] * (1.0 + 0.5 * momenta[w, d])
                        x_trial = x.copy()
                        x_trial[d] += direction * step_size
                        x_trial = np.clip(x_trial, -5, 5)

                        f_trial = func(x_trial)
                        evals += 1

                        if f_trial < best_trial_f:
                            best_trial_f = f_trial
                            best_trial_x = x_trial.copy()

                    if best_trial_f < f:
                        # Update position
                        x = best_trial_x
                        f = best_trial_f
                        walker_improved = True

                        # Update momentum (increase if improving)
                        momenta[w, d] = min(1.0, momenta[w, d] + 0.1)

                        # Update global best
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    else:
                        # No improvement: reduce step size and reset momentum
                        steps[d] *= step_lr
                        momenta[w, d] = 0.0

                # Update walker start position for next iteration
                walker_starts[w] = x.copy()

            # Check for convergence
            if not walker_improved:
                no_improve_count += 1
                # Perturb walkers if stuck
                if no_improve_count >= max_no_improve:
                    for w in range(n_walkers):
                        walker_starts[w] += np.random.normal(0, 0.5, dim)
                        walker_starts[w] = np.clip(walker_starts[w], -5, 5)
                    no_improve_count = 0
                    # Reset step sizes
                    steps = np.full(dim, base_step)
                    momenta = np.zeros((n_walkers, dim))
            else:
                no_improve_count = 0

        # --- Phase 3: Local Refinement with Fine-Grained Coordinate Descent ---
        if evals < budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            step = 0.1
            no_improve = 0
            max_no_improve_cd = 5

            while evals < budget and time.time() < deadline and no_improve < max_no_improve_cd:
                improved = False
                for d in range(dim):
                    if evals >= budget or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        x_trial = x.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        evals += 1
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if step < 1e-6:
                        break
                else:
                    no_improve = 0

            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f
