"""Surrogate-guided optimizer with (1+1)-ES refinement and adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-guided multi-start optimizer with (1+1)-ES and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(7)

        # Track best solution
        best_x = None
        best_f = np.inf
        eval_count = 0

        def check_time():
            return time.time() < deadline - 3.0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        # ===== Phase 1: Structured Initialization with Surrogate Guidance =====
        # Evaluate a diverse set of points to find a good starting basin.
        n_init = min(30, budget - 20)
        init_points = []
        init_points.append(np.zeros(n))

        # Random uniform points in the likely region [-4, 4]
        for _ in range(max(1, n_init // 3)):
            init_points.append(rng.uniform(-4.0, 4.0, n))

        # Closer to center
        for _ in range(max(1, n_init // 3)):
            init_points.append(rng.uniform(-2.0, 2.0, n))

        # Axis-aligned probes (helps if function is separable like Sphere)
        for axis in range(n):
            for val in [-3.0, 3.0, -1.0, 1.0]:
                pt = np.zeros(n)
                pt[axis] = val
                init_points.append(pt)

        # Evaluate initial points
        points = []
        values = []
        for pt in init_points[:n_init]:
            if not check_time() or eval_count >= budget:
                break
            pt = clip_box(pt)
            fv = func(pt)
            try_update(pt, fv)
            points.append(pt)
            values.append(fv)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            try_update(best_x, best_f)
            points.append(best_x)
            values.append(best_f)

        # Fit quadratic surrogate to guide next steps
        X = np.array(points)
        y = np.array(values)

        # Build design matrix for quadratic model
        n_features = 1 + n + n * (n + 1) // 2
        design = np.ones((len(X), n_features))
        for i in range(len(X)):
            x = X[i]
            col = 1
            for d in range(n):
                design[i, col] = x[d]
                col += 1
            for d1 in range(n):
                for d2 in range(d1, n):
                    if d1 == d2:
                        design[i, col] = x[d1] * x[d2]
                    else:
                        design[i, col] = x[d1] * x[d2]
                    col += 1

        # Least squares fit
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        except:
            coeffs = np.zeros(n_features)

        # Extract quadratic form
        a0 = coeffs[0]
        b = coeffs[1:1 + n].copy()
        C_flat = coeffs[1 + n:].copy()

        # Reconstruct C matrix
        C = np.zeros((n, n))
        idx = 0
        for d1 in range(n):
            for d2 in range(d1, n):
                if d1 == d2:
                    C[d1, d2] = C_flat[idx]
                else:
                    C[d1, d2] = C_flat[idx] / 2
                    C[d2, d1] = C_flat[idx] / 2
                idx += 1

        # Ensure C is symmetric
        C = (C + C.T) / 2

        # Find minimum of surrogate
        try:
            if np.linalg.cond(C) < 1e6:
                x_min_surrogate = -np.linalg.solve(C, b / 2)
            else:
                x_min_surrogate = -np.linalg.pinv(C) @ (b / 2)
            x_min_surrogate = np.clip(x_min_surrogate, -4.0, 4.0)
        except:
            x_min_surrogate = best_x.copy()

        # Evaluate the surrogate minimum
        if check_time() and eval_count < budget:
            x_test = clip_box(x_min_surrogate)
            f_test = func(x_test)
            try_update(x_test, f_test)
            if f_test < best_f:
                best_x = x_test.copy()

        # ===== Phase 2: Surrogate-Guided (1+1)-ES with Adaptive Restarts =====
        # Use (1+1)-ES for fast local refinement, with surrogate-guided restarts
        x_local = best_x.copy()
        f_local = best_f
        sigma = 0.5
        no_improve = 0
        max_no_improve = 30
        restart_count = 0
        max_restarts = 10

        while check_time() and eval_count < budget:
            # Generate one offspring
            if not check_time() or eval_count >= budget:
                break

            # Generate step direction
            z = rng.standard_normal(n)

            # If we have a good surrogate, bias the step toward the surrogate minimum
            if restart_count == 0 and eval_count < 50:
                # Early on, use surrogate gradient to guide direction
                try:
                    grad_surrogate = 2 * C @ x_local + b
                    grad_norm = np.linalg.norm(grad_surrogate)
                    if grad_norm > 1e-8:
                        # Blend random direction with surrogate gradient direction
                        z = 0.7 * z - 0.3 * (grad_surrogate / grad_norm) * np.linalg.norm(z)
                except:
                    pass

            x_try = clip_box(x_local + sigma * z)
            f_try = func(x_try)
            try_update(x_try, f_try)

            # Accept if better
            if f_try < f_local - 1e-12:
                x_local = x_try
                f_local = f_try
                no_improve = 0
                sigma = min(sigma * 1.05, 2.0)
            else:
                no_improve += 1
                sigma = max(sigma * 0.95, 0.01)

            # Check for global best update
            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

            # Restart if stagnated
            if no_improve > max_no_improve:
                restart_count += 1
                if restart_count > max_restarts:
                    break

                # Try to find a new basin using the surrogate or random
                if restart_count <= 3:
                    # Use surrogate to suggest a new starting point
                    try:
                        # Sample around the surrogate minimum with some noise
                        x_restart = x_min_surrogate + rng.normal(0, 1.0, n)
                        x_restart = clip_box(x_restart)
                    except:
                        x_restart = rng.uniform(-4.0, 4.0, n)
                else:
                    # Random restart
                    x_restart = rng.uniform(-4.0, 4.0, n)

                f_restart = func(x_restart)
                try_update(x_restart, f_restart)

                if f_restart < f_local:
                    x_local = x_restart
                    f_local = f_restart
                else:
                    # Keep current best local, but reset stagnation counter
                    x_local = best_x.copy()
                    f_local = best_f

                no_improve = 0
                sigma = 1.0
                max_no_improve = min(max_no_improve + 5, 50)

            # Early termination if converged
            if sigma < 1e-5 and no_improve > 10:
                break

        # ===== Phase 3: Fine-tuning with Very Small Steps =====
        # If we have budget left, do a final polish with tiny steps
        if check_time() and eval_count < budget:
            sigma_fine = max(0.01, sigma * 0.1)
            x_fine = best_x.copy()
            f_fine = best_f
            no_improve_fine = 0
            max_fine_steps = min(30, budget - eval_count)

            for _ in range(max_fine_steps):
                if not check_time() or eval_count >= budget:
                    break
                y = clip_box(x_fine + sigma_fine * rng.standard_normal(n))
                fy = func(y)
                try_update(y, fy)

                if fy < f_fine - 1e-12:
                    x_fine = y
                    f_fine = fy
                    no_improve_fine = 0
                    sigma_fine = min(sigma_fine * 1.02, 0.1)
                else:
                    sigma_fine *= 0.95
                    no_improve_fine += 1

                if sigma_fine < 1e-6 or no_improve_fine > 15:
                    break

            if f_fine < best_f:
                best_x = x_fine
                best_f = f_fine

        return best_x, best_f
# EVOLVE-BLOCK-END
