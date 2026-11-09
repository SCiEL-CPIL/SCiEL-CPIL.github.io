"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+λ)-ES with paraboloid fitting and restarts for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def fit_paraboloid(center, points, values):
            """Fit a quadratic model to points and find its minimum."""
            # points: list of (n,) arrays, values: list of floats
            # Fit f(x) ≈ c + g^T(x-c) + 0.5*(x-c)^T H (x-c)
            # Using 2n+1 points: center, center ± h*e_i
            m = len(points)
            if m < n + 1:
                return None
            # Build design matrix for quadratic: [1, (x-c)_i, (x-c)_i*(x-c)_j for i<=j]
            # Simpler: use least squares on 2n+1 points
            X = []
            y = []
            for p, v in zip(points, values):
                d = p - center
                row = [1.0]
                row.extend(d)
                for i in range(n):
                    for j in range(i, n):
                        row.append(d[i] * d[j])
                X.append(row)
                y.append(v)
            X = np.array(X)
            y = np.array(y)
            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                return None
            # Extract Hessian and gradient from coeffs
            # coeffs[0] = c, coeffs[1:n+1] = g, coeffs[n+1:] = quadratic terms
            c = coeffs[0]
            g = coeffs[1:n+1].copy()
            H = np.zeros((n, n))
            idx = n + 1
            for i in range(n):
                for j in range(i, n):
                    hval = coeffs[idx]
                    idx += 1
                    H[i, j] = hval
                    H[j, i] = hval
                # Diagonal: the coefficient for d_i^2 is H[i,i], but in our parametrization
                # we have 0.5*(x-c)^T H (x-c), so the coefficient of d_i^2 is H[i,i]
                # and coefficient of d_i*d_j (i≠j) is H[i,j]
                # But in our design matrix, for i=j we have d_i^2, for i<j we have d_i*d_j
                # So coeffs for d_i^2 should be H[i,i], and for d_i*d_j should be H[i,j]
                # However, 0.5*(x-c)^T H (x-c) = 0.5 * sum_i H[i,i] d_i^2 + sum_{i<j} H[i,j] d_i d_j
                # So the coefficient of d_i^2 in the expansion is 0.5*H[i,i], and d_i*d_j is H[i,j]
                # Our design matrix uses d_i^2 and d_i*d_j directly, so:
                # coeffs[idx] for d_i^2 = 0.5 * H[i,i]
                # coeffs[idx] for d_i*d_j = H[i,j]
                # Let's fix this:
            H2 = np.zeros((n, n))
            idx = n + 1
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        H2[i, i] = 2.0 * coeffs[idx]  # because 0.5*H[i,i] = coeff => H[i,i] = 2*coeff
                    else:
                        H2[i, j] = coeffs[idx]
                        H2[j, i] = coeffs[idx]
                    idx += 1
            H = H2
            # Solve H * dx = -g
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                except:
                    return None
            # Check that H is positive definite (or at least the step reduces the model)
            model_at_center = c
            model_at_new = c + g @ dx + 0.5 * dx @ H @ dx
            if model_at_new >= model_at_center:
                return None
            new_x = center + dx
            new_x = np.clip(new_x, -5, 5)
            return new_x

        # Phase 1: Coarse search with (1+λ)-ES and multiple restarts
        # Phase 2: Paraboloid fitting around best point to jump to basin center

        # Restart strategy: 6 restarts
        # First 4 are exploratory, last 2 are exploitative around best
        num_restarts = 6
        budget_fracs = [0.20, 0.15, 0.15, 0.15, 0.20, 0.15]

        for restart_idx in range(num_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            budget_this_restart = int(total_budget * budget_fracs[restart_idx])
            budget_this_restart = min(budget_this_restart, total_budget - calls_made)
            if budget_this_restart < 15:
                continue

            # Initialization
            if restart_idx == 0:
                x = np.zeros(n)
            elif restart_idx == 1:
                x = rng.uniform(-4, 4, n)
            elif restart_idx == 2:
                x = rng.uniform(-5, 5, n)
            elif restart_idx == 3:
                x = rng.uniform(-5, 5, n)
            elif restart_idx == 4:
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 2.0, -5, 5)
                else:
                    x = rng.uniform(-5, 5, n)
            else:
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 1.0, -5, 5)
                else:
                    x = rng.uniform(-5, 5, n)

            fx = evaluate(x)

            # (1+λ)-ES parameters
            lam = 6
            sigma = 1.5
            steps = 0
            max_steps = budget_this_restart
            no_improve = 0
            max_no_improve = 30

            while calls_made < total_budget and steps < max_steps:
                if time.time() > deadline:
                    break

                # Generate λ offspring
                offspring = []
                offspring_f = []
                for i in range(lam):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * z
                    fy = evaluate(y)
                    offspring.append(y)
                    offspring_f.append(fy)

                if not offspring:
                    break

                # Select best offspring
                best_idx = np.argmin(offspring_f)
                best_y = offspring[best_idx]
                best_yf = offspring_f[best_idx]

                if best_yf < fx:
                    x = best_y
                    fx = best_yf
                    no_improve = 0
                    # Adapt sigma: increase on success
                    sigma *= 1.2
                else:
                    no_improve += 1
                    sigma *= 0.8

                sigma = np.clip(sigma, 1e-6, 5.0)
                steps += 1

                if no_improve >= max_no_improve:
                    break

            # After ES, try paraboloid fit around current best
            if calls_made < total_budget and time.time() < deadline:
                # Collect points for paraboloid fit
                h = 0.5
                center = x.copy()
                points = [center]
                values = [fx]
                for i in range(n):
                    for sign in [1, -1]:
                        p = center.copy()
                        p[i] += sign * h
                        p = np.clip(p, -5, 5)
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        v = evaluate(p)
                        points.append(p)
                        values.append(v)
                # Fit and jump
                jump = fit_paraboloid(center, points, values)
                if jump is not None:
                    # Evaluate the jump
                    if calls_made < total_budget and time.time() < deadline:
                        fj = evaluate(jump)
                        # If jump was good, do a local refinement
                        if fj < fx:
                            x = jump
                            fx = fj
                            # Refine with smaller steps
                            sigma = 0.3
                            for _ in range(20):
                                if calls_made >= total_budget or time.time() > deadline:
                                    break
                                z = rng.standard_normal(n)
                                y = x + sigma * z
                                fy = evaluate(y)
                                if fy < fx:
                                    x = y
                                    fx = fy
                                    sigma *= 1.1
                                else:
                                    sigma *= 0.8
                                sigma = np.clip(sigma, 1e-6, 1.0)

        # Final paraboloid fit around global best
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            h = 0.3
            center = best_x.copy()
            points = [center]
            values = [best_f]
            for i in range(n):
                for sign in [1, -1]:
                    p = center.copy()
                    p[i] += sign * h
                    p = np.clip(p, -5, 5)
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    v = evaluate(p)
                    points.append(p)
                    values.append(v)
            jump = fit_paraboloid(center, points, values)
            if jump is not None:
                evaluate(jump)

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
