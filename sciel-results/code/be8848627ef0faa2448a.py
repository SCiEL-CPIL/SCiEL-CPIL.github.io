import numpy as np
import time

class Optimizer:
    """
    Trust-region optimizer using quadratic model fitting for BBOB f13 (Sharp Ridge).
    Uses local surrogate models to capture anisotropic curvature.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        
        best_x = None
        best_f = float('inf')
        
        calls = 0
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                return func(np.asarray(x, dtype=np.float64))
            except:
                return None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=np.float64).copy()

        def fit_quadratic_model(points, values, center):
            """
            Fit a quadratic model: f(x) ≈ c + g^T (x-center) + 0.5 (x-center)^T H (x-center)
            points: (n, dim) array
            values: (n,) array
            center: (dim,) array
            Returns: (c, g, H) where H is symmetric (dim, dim)
            """
            n = len(points)
            if n < dim + 1:
                # Not enough points, return a flat model
                return values[0], np.zeros(dim), np.zeros((dim, dim))
            
            # Compute centered points
            X = points - center  # (n, dim)
            
            # Build design matrix for quadratic terms
            # Terms: 1, x_i, x_i*x_j for i<=j
            # Linear terms: dim
            # Quadratic terms: dim*(dim+1)/2
            n_lin = dim
            n_quad = dim * (dim + 1) // 2
            n_params = 1 + n_lin + n_quad
            
            if n < n_params:
                # Underdetermined, use least squares with regularization
                pass
            
            # Build the design matrix
            A = np.zeros((n, n_params))
            A[:, 0] = 1.0  # constant term
            
            # Linear terms
            A[:, 1:1+n_lin] = X
            
            # Quadratic terms
            idx = 1 + n_lin
            for i in range(dim):
                for j in range(i, dim):
                    A[:, idx] = X[:, i] * X[:, j]
                    idx += 1
            
            # Solve least squares with small regularization for stability
            try:
                # Add small regularization
                reg = 1e-10
                A_reg = A.copy()
                A_reg[:, 1:] += 0  # don't regularize constant
                # Use np.linalg.lstsq
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, values, rcond=1e-10)
            except:
                return values[0], np.zeros(dim), np.zeros((dim, dim))
            
            # Extract coefficients
            c = coeffs[0]
            g = coeffs[1:1+n_lin].copy()
            
            # Reconstruct H from quadratic coefficients
            H = np.zeros((dim, dim))
            idx = 1 + n_lin
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        H[i, i] = coeffs[idx]
                    else:
                        H[i, j] = coeffs[idx]
                        H[j, i] = coeffs[idx]
                    idx += 1
            
            # Ensure H is symmetric
            H = 0.5 * (H + H.T)
            
            return c, g, H

        def minimize_quadratic(g, H, center, delta_max, box_lo, box_hi):
            """
            Minimize the quadratic model c + g^T d + 0.5 d^T H d
            subject to ||d|| <= delta_max and box constraints.
            Returns the step d.
            """
            # If H is near zero, take a step along -g
            h_norm = np.linalg.norm(H, 'fro')
            if h_norm < 1e-12:
                # Linear problem: minimize g^T d s.t. ||d|| <= delta_max
                g_norm = np.linalg.norm(g)
                if g_norm < 1e-15:
                    return np.zeros(dim)
                d = -g / g_norm * delta_max
                # Check box constraints
                for i in range(dim):
                    if center[i] + d[i] < box_lo[i]:
                        d[i] = box_lo[i] - center[i]
                    elif center[i] + d[i] > box_hi[i]:
                        d[i] = box_hi[i] - center[i]
                return d
            
            # Use trust-region subproblem solver
            # Simple approach: use scipy's trust region or implement a basic one
            # For simplicity, use a projected gradient step with line search on the model
            
            # Try to find the unconstrained minimizer first
            try:
                # H may be singular, use pseudo-inverse
                H_inv = np.linalg.pinv(H)
                d_uncon = -H_inv @ g
            except:
                d_uncon = -g / max(np.linalg.norm(g), 1e-15) * delta_max
            
            # Project onto trust region
            d_norm = np.linalg.norm(d_uncon)
            if d_norm > delta_max:
                d_uncon = d_uncon / d_norm * delta_max
            
            # Apply box constraints via simple projection
            d = d_uncon.copy()
            for i in range(dim):
                if center[i] + d[i] < box_lo[i]:
                    d[i] = box_lo[i] - center[i]
                elif center[i] + d[i] > box_hi[i]:
                    d[i] = box_hi[i] - center[i]
            
            # Refine with a few projected gradient steps on the model
            for _ in range(10):
                grad_model = g + H @ d
                # Projected gradient descent step
                step_size = delta_max / max(np.linalg.norm(grad_model), 1e-15)
                d_new = d - step_size * grad_model
                for i in range(dim):
                    if center[i] + d_new[i] < box_lo[i]:
                        d_new[i] = box_lo[i] - center[i]
                    elif center[i] + d_new[i] > box_hi[i]:
                        d_new[i] = box_hi[i] - center[i]
                # Check if we're still in the trust region
                if np.linalg.norm(d_new) > delta_max:
                    d_new = d_new / np.linalg.norm(d_new) * delta_max
                d = d_new
            
            return d

        # Phase 1: Global exploration with adaptive sampling
        # Start with a moderate number of random samples, then switch to local refinement
        n_global = min(30, budget // 10)
        
        box_lo = np.full(dim, -5.0)
        box_hi = np.full(dim, 5.0)
        
        for i in range(n_global):
            x = rng.uniform(-5, 5, size=dim)
            f = safe_call(x)
            if f is None:
                break
            update_best(x, f)
        
        if best_x is None:
            return best_x, best_f
        
        # Phase 2: Trust-region local refinement
        # Maintain a set of points around the current best to build the model
        # Start with a simplex-like set of points
        
        # Initialize the point set with best_x and neighbors
        points = [best_x.copy()]
        values = [best_f]
        
        # Add neighbors along each axis
        initial_step = 1.0
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            xi = best_x.copy()
            xi[i] += initial_step
            fi = safe_call(xi)
            if fi is None:
                break
            points.append(xi.copy())
            values.append(fi)
            update_best(xi, fi)
            
            # Also add the negative direction
            if calls < budget and time.time() < deadline:
                xi_neg = best_x.copy()
                xi_neg[i] -= initial_step
                fi_neg = safe_call(xi_neg)
                if fi_neg is None:
                    break
                points.append(xi_neg.copy())
                values.append(fi_neg)
                update_best(xi_neg, fi_neg)
        
        if len(points) < 2:
            return best_x, best_f
        
        # Trust-region parameters
        delta = 1.0  # initial trust region radius
        delta_max = 4.0  # maximum trust region radius
        delta_min = 1e-4  # minimum trust region radius
        
        # Main trust-region loop
        while calls < budget and time.time() < deadline:
            # Fit quadratic model
            points_arr = np.array(points)
            values_arr = np.array(values)
            
            # Use the best point as center
            best_idx = np.argmin(values_arr)
            center = points_arr[best_idx].copy()
            
            c, g, H = fit_quadratic_model(points_arr, values_arr, center)
            
            # Find the step using the trust-region subproblem
            d = minimize_quadratic(g, H, center, delta, box_lo, box_hi)
            
            # Evaluate at the new point
            x_new = center + d
            x_new = np.clip(x_new, -5, 5)
            f_new = safe_call(x_new)
            if f_new is None:
                break
            
            # Update trust region
            if f_new < values_arr[best_idx]:
                # Good step, increase trust region
                delta = min(delta * 2.0, delta_max)
                # Accept the new point
                points.append(x_new.copy())
                values.append(f_new)
                update_best(x_new, f_new)
                
                # Remove the worst point to keep the set manageable
                if len(points) > 2 * dim + 2:
                    worst_idx = np.argmax(values)
                    points.pop(worst_idx)
                    values.pop(worst_idx)
            else:
                # Bad step, decrease trust region
                delta = max(delta * 0.5, delta_min)
            
            # Check for convergence
            if delta < delta_min * 1.1:
                # Try a random restart from the best point
                if calls < budget and time.time() < deadline:
                    # Generate a new random point near the best
                    x_restart = best_x + rng.normal(0, 0.5, size=dim)
                    x_restart = np.clip(x_restart, -5, 5)
                    f_restart = safe_call(x_restart)
                    if f_restart is None:
                        break
                    update_best(x_restart, f_restart)
                    
                    # Reset the point set
                    points = [best_x.copy()]
                    values = [best_f]
                    for i in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        xi = best_x.copy()
                        xi[i] += 0.5
                        fi = safe_call(xi)
                        if fi is None:
                            break
                        points.append(xi.copy())
                        values.append(fi)
                        update_best(xi, fi)
                    
                    delta = 0.5  # reset trust region
        
        return best_x, best_f
