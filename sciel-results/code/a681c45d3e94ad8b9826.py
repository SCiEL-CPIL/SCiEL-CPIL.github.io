"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: random spread + CMA-ES + quadratic polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Quick random sampling to find a good starting point (50 calls)
        n_init = 50
        x = np.zeros((n_init, dim))
        fx = np.zeros(n_init)
        for i in range(n_init):
            x[i] = rng.uniform(-5, 5, dim)
            fx[i] = func(x[i])
            if time.time() > deadline:
                break

        best_idx = np.argmin(fx)
        x_best = x[best_idx].copy()
        f_best = fx[best_idx]

        # Phase 2: CMA-ES for the bulk of the budget
        # Adapt sigma based on initial spread
        sigma = 2.0
        lambda_ = 4  # population size
        calls_made = n_init

        for gen in range(100):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Generate lambda offspring
            for i in range(lambda_):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = x_best + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < f_best:
                    x_best = y
                    f_best = fy

            # Update sigma (simple 1/5 rule approximation)
            # In a full CMA-ES we'd track success rate; here we use a heuristic
            sigma *= 0.95  # gentle decrease

        # Phase 3: Local quadratic model fitting for super-linear convergence
        # Use remaining budget for local search around x_best
        remaining = budget - calls_made
        if remaining > 20 and time.time() < deadline:
            # Sample points around x_best to fit a quadratic model
            n_local = min(remaining, 40)
            local_x = np.zeros((n_local, dim))
            local_f = np.zeros(n_local)
            
            # First point is x_best
            local_x[0] = x_best
            local_f[0] = f_best
            
            # Generate local samples
            local_sigma = 0.5
            for i in range(1, n_local):
                if time.time() > deadline:
                    break
                dx = local_sigma * rng.standard_normal(dim)
                xi = np.clip(x_best + dx, -5, 5)
                local_x[i] = xi
                local_f[i] = func(xi)
                calls_made += 1
                
                # Adapt local sigma
                if local_f[i] < f_best:
                    f_best = local_f[i]
                    x_best = xi
                    local_sigma *= 0.8
                else:
                    local_sigma *= 0.7

            # Fit centered quadratic model: f(x) ≈ f0 + g^T(x-x0) + (x-x0)^T H (x-x0)
            # Use the local samples to estimate gradient and Hessian
            x0 = x_best
            f0 = f_best
            
            # Center the data
            X = local_x - x0
            F = local_f - f0
            
            # Solve for linear and quadratic terms using least squares
            # Build design matrix: [X, X^2]
            n_pts = len(F)
            n_params = dim + dim * (dim + 1) // 2  # gradient + symmetric Hessian
            
            # Only fit if we have enough points
            if n_pts >= n_params + 1:
                A = np.zeros((n_pts, n_params))
                for i in range(n_pts):
                    xi = X[i]
                    # Linear terms
                    A[i, :dim] = xi
                    # Quadratic terms (symmetric)
                    quad_idx = dim
                    for a in range(dim):
                        for b in range(a, dim):
                            A[i, quad_idx] = xi[a] * xi[b]
                            quad_idx += 1
                
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(A, F, rcond=None)
                    if rank >= n_params - 1:  # sufficient rank
                        g = coeffs[:dim]
                        h_coeffs = coeffs[dim:]
                        
                        # Reconstruct Hessian
                        H = np.zeros((dim, dim))
                        quad_idx = 0
                        for a in range(dim):
                            for b in range(a, dim):
                                H[a, b] = h_coeffs[quad_idx]
                                if a != b:
                                    H[b, a] = h_coeffs[quad_idx]
                                quad_idx += 1
                        
                        # Newton step: solve H * dx = -g
                        try:
                            # Add regularization if H is ill-conditioned
                            H_reg = H + 1e-8 * np.eye(dim)
                            dx = np.linalg.solve(H_reg, -g)
                            # Limit step size
                            step_norm = np.linalg.norm(dx)
                            if step_norm > 2.0:
                                dx = dx / step_norm * 2.0
                            
                            x_new = np.clip(x_best + dx, -5, 5)
                            f_new = func(x_new)
                            calls_made += 1
                            if f_new < f_best:
                                x_best = x_new
                                f_best = f_new
                        except np.linalg.LinAlgError:
                            pass
                except np.linalg.LinAlgError:
                    pass

        return x_best, f_best
# EVOLVE-BLOCK-END
