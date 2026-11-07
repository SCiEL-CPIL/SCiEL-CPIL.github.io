"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free trust-region optimizer with quadratic model fitting."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        def safe_call(x):
            """Call func with bounds clipping and error handling."""
            x = np.clip(x, -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return np.inf

        # --- Phase 1: Initial sampling to build a model ---
        # We'll use a trust-region approach with a quadratic model.
        # Start with a reasonable initial point and expand the trust region.
        
        rng = np.random.default_rng(123)
        
        # Initial point: try center and a few random points
        x0 = np.zeros(n)
        fx0 = safe_call(x0)
        
        # Build initial set of points around x0 for model fitting
        # Use a sphere of points to estimate gradient and Hessian
        initial_radius = 1.0
        points = [x0.copy()]
        values = [fx0]
        
        # Add axis-aligned points
        for i in range(n):
            for sign in [1, -1]:
                x_new = x0.copy()
                x_new[i] += sign * initial_radius
                x_new = np.clip(x_new, -5.0, 5.0)
                points.append(x_new.copy())
                values.append(safe_call(x_new))
        
        # Add some diagonal points for better curvature estimation
        for i in range(n):
            for j in range(i + 1, n):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        x_new = x0.copy()
                        x_new[i] += s1 * initial_radius * 0.7
                        x_new[j] += s2 * initial_radius * 0.7
                        x_new = np.clip(x_new, -5.0, 5.0)
                        points.append(x_new.copy())
                        values.append(safe_call(x_new))
        
        calls_made = len(points)
        if calls_made >= total_budget:
            best_idx = np.argmin(values)
            return points[best_idx], values[best_idx]

        # --- Phase 2: Trust-region iterations with quadratic model ---
        best_x = points[np.argmin(values)].copy()
        best_f = min(values)
        
        # Build quadratic model from points
        def fit_quadratic_model(pts, vals):
            """Fit a quadratic model f(x) ≈ f0 + g^T(x-x0) + 0.5*(x-x0)^T H (x-x0).
            Returns (f0, g, H) where x0 is the first point."""
            x0 = pts[0]
            d = len(pts[0])
            m = len(pts)
            
            # Build design matrix for quadratic terms
            # Variables: f0, g (d), H (d*d/2 unique elements)
            # Total params: 1 + d + d*(d+1)/2
            n_params = 1 + d + d * (d + 1) // 2
            
            # For each point, compute [1, (x-x0), quadratic terms]
            A = np.zeros((m, n_params))
            A[:, 0] = 1.0
            A[:, 1:1+d] = pts - x0  # linear terms
            
            # Quadratic terms: 0.5 * x^T H x, but we use the form where
            # the quadratic part is sum_{i<=j} H_{ij} * x_i * x_j
            # Actually, let's use the standard form:
            # f(x) = f0 + g^T d + 0.5 d^T H d where d = x - x0
            # So the quadratic terms are 0.5 * sum_{i,j} H_{ij} d_i d_j
            # We can write this as sum_{i<=j} c_{ij} d_i d_j where
            # c_{ii} = 0.5*H_{ii} and c_{ij} = 0.5*H_{ij} for i != j
            # But it's easier to just use the full symmetric H.
            
            # Let's use a simpler approach: fit f(x) = b + L^T x + 0.5 x^T Q x
            # where Q is symmetric. This has 1 + d + d*(d+1)/2 parameters.
            
            # For point x: [1, x_1, ..., x_d, x_1^2, x_1*x_2, ..., x_d^2]
            quad_cols = 0
            for i in range(d):
                for j in range(i, d):
                    if i == j:
                        A[:, 1+d+quad_cols] = pts[:, i] ** 2
                    else:
                        A[:, 1+d+quad_cols] = 2.0 * pts[:, i] * pts[:, j]
                    quad_cols += 1
            
            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, vals, rcond=None)
            except np.linalg.LinAlgError:
                # Fallback: just use the best point as the model
                return vals[0], np.zeros(d), np.zeros((d, d))
            
            f0 = coeffs[0]
            g = coeffs[1:1+d]
            
            # Reconstruct Q from quadratic coefficients
            Q = np.zeros((d, d))
            idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    if i == j:
                        Q[i, i] = coeffs[idx]
                    else:
                        Q[i, j] = coeffs[idx] / 2.0
                        Q[j, i] = coeffs[idx] / 2.0
                    idx += 1
            
            return f0, g, Q

        def solve_subproblem(x0, f0, g, Q, radius):
            """Solve the trust-region subproblem:
            min g^T d + 0.5 d^T Q d
            s.t. ||d|| <= radius
            
            Uses a simple gradient descent with projection for small problems.
            """
            d = np.zeros_like(x0)
            lr = 1.0 / (np.linalg.norm(Q, ord=2) + 1e-8)
            lr = min(lr, 0.5)
            
            for _ in range(200):
                grad = g + Q @ d
                d_new = d - lr * grad
                # Project onto trust region
                norm = np.linalg.norm(d_new)
                if norm > radius:
                    d_new = d_new * (radius / norm)
                if np.linalg.norm(d_new - d) < 1e-10:
                    break
                d = d_new
            
            # Check if we're at the boundary; if so, try a few more iterations
            # with a smaller learning rate
            norm = np.linalg.norm(d)
            if abs(norm - radius) < 1e-6:
                lr2 = lr * 0.1
                for _ in range(100):
                    grad = g + Q @ d
                    d_new = d - lr2 * grad
                    norm = np.linalg.norm(d_new)
                    if norm > radius:
                        d_new = d_new * (radius / norm)
                    if np.linalg.norm(d_new - d) < 1e-10:
                        break
                    d = d_new
            
            return d

        # Trust region radius
        tr_radius = initial_radius * 1.5
        min_radius = 1e-4
        max_radius = 4.0
        stagnation_count = 0
        max_stagnation = 10

        while calls_made < total_budget and time.time() < deadline:
            # Fit quadratic model using recent points
            # Use all points for better model, but limit to last 50 for speed
            recent_pts = points[-50:] if len(points) > 50 else points
            recent_vals = values[-50:] if len(values) > 50 else values
            
            if len(recent_pts) < n + 2:
                # Not enough points, add more
                x_new = best_x + tr_radius * rng.standard_normal(n)
                x_new = np.clip(x_new, -5.0, 5.0)
                points.append(x_new.copy())
                values.append(safe_call(x_new))
                calls_made += 1
                continue
            
            f0, g, Q = fit_quadratic_model(recent_pts, recent_vals)
            
            # Solve trust-region subproblem
            d = solve_subproblem(best_x, f0, g, Q, tr_radius)
            x_new = np.clip(best_x + d, -5.0, 5.0)
            
            # Evaluate
            f_new = safe_call(x_new)
            calls_made += 1
            points.append(x_new.copy())
            values.append(f_new)
            
            # Accept or reject
            predicted_reduction = f0 - (f0 + g @ d + 0.5 * d @ Q @ d)
            actual_reduction = best_f - f_new
            
            if actual_reduction > 0:
                # Accept
                best_x = x_new.copy()
                best_f = f_new
                stagnation_count = 0
                # Adjust trust region
                rho = actual_reduction / max(predicted_reduction, 1e-12)
                if rho > 0.75:
                    tr_radius = min(tr_radius * 1.5, max_radius)
                elif rho < 0.25:
                    tr_radius = max(tr_radius * 0.5, min_radius)
            else:
                # Reject
                stagnation_count += 1
                tr_radius = max(tr_radius * 0.5, min_radius)
                
                if stagnation_count >= max_stagnation:
                    # Restart with a new random point
                    x_new = rng.uniform(-4, 4, n)
                    f_new = safe_call(x_new)
                    calls_made += 1
                    points.append(x_new.copy())
                    values.append(f_new)
                    if f_new < best_f:
                        best_x = x_new.copy()
                        best_f = f_new
                    tr_radius = 1.0
                    stagnation_count = 0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
