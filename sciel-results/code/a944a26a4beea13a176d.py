"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """
    Hybrid strategy:
    1. Sample integer lattice points (Rastrigin minima are at integers pre-rotation).
    2. Fit a sparse Fourier surrogate to the sampled landscape.
    3. Use surrogate gradient descent for zero-cost refinement steps.
    4. Use remaining budget for Nelder-Mead local polish on the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() >= deadline:
                return np.inf
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Sample integer lattice points
        # Rastrigin global minima are at integer coordinates.
        # After rotation/shift, they move, but nearby integers are still good basins.
        # Sample a subset of integers in [-4, 4]^5.
        n_grid = 40
        grid_points = []
        for _ in range(n_grid):
            pt = rng.integers(-4, 5, size=dim).astype(float)
            grid_points.append(pt)

        # Evaluate grid points
        grid_values = []
        for pt in grid_points:
            if calls >= budget or time.time() >= deadline:
                break
            val = func(pt)
            grid_values.append((val, pt))
            calls += 1

        if not grid_values:
            # Fallback: random search
            x = rng.uniform(-5, 5, size=dim)
            fx = wrapper(x)
            sigma = 2.0
            for _ in range(budget - calls):
                if time.time() >= deadline or calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = wrapper(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
            return x, fx

        # Sort by value, take top candidates
        grid_values.sort(key=lambda t: t[0])
        top_candidates = grid_values[:min(5, len(grid_values))]

        # Phase 2: Build Fourier surrogate from sampled points
        # Use all grid points for fitting
        X = np.array([pt for _, pt in grid_values])
        Y = np.array([val for val, _ in grid_values])

        # Build Fourier basis: sin/cos of k * x_i for small k
        # Keep it sparse to avoid overfitting with ~40 points in 5D
        max_freq = 2
        features = []
        feature_names = []
        for i in range(dim):
            for k in range(1, max_freq + 1):
                features.append(np.sin(k * X[:, i]))
                features.append(np.cos(k * X[:, i]))
                feature_names.append(f"sin_{k}_{i}")
                feature_names.append(f"cos_{k}_{i}")

        if features:
            X_feat = np.column_stack(features)
            # Add bias
            X_feat = np.column_stack([np.ones(len(X_feat)), X_feat])

            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(X_feat, Y, rcond=None)
            except Exception:
                coeffs = None

            if coeffs is not None:
                # Phase 3: Surrogate gradient descent from best grid point
                x = top_candidates[0][1].copy()
                fx = top_candidates[0][0]

                # Compute surrogate gradient
                def surrogate_grad(x_cur):
                    grad = np.zeros(dim)
                    for i in range(dim):
                        for k in range(1, max_freq + 1):
                            # d/dx_i sin(k*x_i) = k*cos(k*x_i)
                            # d/dx_i cos(k*x_i) = -k*sin(k*x_i)
                            idx_sin = 1 + dim * max_freq + i * max_freq + (k - 1) * 2
                            idx_cos = idx_sin + 1
                            grad[i] += coeffs[idx_sin] * k * np.cos(k * x_cur[i])
                            grad[i] -= coeffs[idx_cos] * k * np.sin(k * x_cur[i])
                    return grad

                # Take a few gradient steps (zero-cost)
                lr = 0.1
                for _ in range(50):
                    if time.time() >= deadline:
                        break
                    g = surrogate_grad(x)
                    x_new = np.clip(x - lr * g, -5, 5)
                    # Only accept if within bounds (surrogate is only valid near samples)
                    x = x_new

                # Evaluate the surrogate-guided point
                f_new = wrapper(x)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x.copy()

        # Phase 4: Nelder-Mead local polish from best point
        if best_x is not None:
            remaining_budget = budget - calls
            if remaining_budget > 10:
                nm_maxiter = max(1, remaining_budget // (dim + 1))
                try:
                    res_nm = scipy.optimize.minimize(
                        wrapper,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': nm_maxiter, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                except Exception:
                    pass

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapper(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
