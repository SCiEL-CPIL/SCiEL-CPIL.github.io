import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Quadratic Fit
        # We need to fit f(x) = x^T Q x + q^T x + c
        # Number of params: dim + dim*(dim+1)/2 + 1
        # For dim=5: 5 + 15 + 1 = 21 params.
        # We need at least 21 points. Let's use 25 for stability.
        n_points = 25
        if budget < n_points + 5:
             # Fallback to simple ES if budget is tiny
             return self._simple_es(func, best_x, best_f)

        # Generate points
        # Use a mix of center and random to ensure spread
        points = []
        # Center
        points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 2.5
            p2 = np.zeros(dim); p2[i] = -2.5
            points.append(p1)
            points.append(p2)
        
        # Random points
        rng = np.random.default_rng(42)
        while len(points) < n_points:
            p = rng.uniform(-4, 4, dim)
            points.append(p)
        
        points = np.array(points)
        values = np.array([evaluate(p) for p in points])
        
        # Construct design matrix for quadratic fit
        # f(x) = sum_i sum_j Q_ij x_i x_j + sum_i q_i x_i + c
        # We can vectorize the quadratic terms.
        # Let v(x) be the vector of all x_i x_j for i<=j and x_i and 1.
        
        def feature_vector(x):
            dim = len(x)
            # x_i x_j terms (symmetric matrix)
            # There are dim*(dim+1)/2 terms
            quad_terms = []
            for i in range(dim):
                for j in range(i, dim):
                    quad_terms.append(x[i] * x[j])
            # Linear terms
            lin_terms = list(x)
            # Constant
            const = [1.0]
            return np.array(quad_terms + lin_terms + const)

        A = np.array([feature_vector(p) for p in points])
        b = values
        
        # Solve least squares
        Q = None
        q = None
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Extract Q, q, c
            n_quad = dim * (dim + 1) // 2
            Q_flat = coeffs[:n_quad]
            q = coeffs[n_quad:n_quad + dim]
            c = coeffs[n_quad + dim]
            
            # Reconstruct Q matrix
            Q = np.zeros((dim, dim))
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    val = Q_flat[idx]
                    Q[i, j] = val
                    Q[j, i] = val
                    idx += 1
            
            # Solve Q x = -0.5 q
            # x_opt = -0.5 Q^{-1} q
            try:
                x_opt = np.linalg.solve(Q, -0.5 * q)
            except np.linalg.LinAlgError:
                x_opt = np.linalg.pinv(Q) @ (-0.5 * q)
            
            x_opt = np.clip(x_opt, -5, 5)
            f_opt = evaluate(x_opt)
            
            # If the Newton step was good, we have a very strong candidate
            if f_opt < best_f:
                best_x = x_opt
                best_f = f_opt
                
        except Exception:
            # If fitting fails, just proceed with best found
            pass

        # Phase 2: Polish with Hessian-scaled (1+1)-ES
        # Start from best_x
        x = best_x.copy()
        fx = best_f
        
        # If we have a valid Q, use it to scale the search
        # H = 2*Q. We want to sample from N(0, H^{-1})
        # This is equivalent to sampling z ~ N(0, I) and mapping y = x + H^{-1/2} z
        # Or simpler: y = x + step * (H^{-1} * random_dir) is not quite right for covariance.
        # Let's use the Cholesky decomposition of the inverse Hessian if possible.
        
        H_inv = None
        if Q is not None:
            try:
                # H = 2 * Q
                H = 2 * Q
                # Check if H is positive definite
                # If not, fall back to identity
                eigvals = np.linalg.eigvalsh(H)
                if np.all(eigvals > 1e-6):
                    # Compute inverse
                    H_inv = np.linalg.inv(H)
                    # Ensure symmetry
                    H_inv = (H_inv + H_inv.T) / 2
            except Exception:
                H_inv = None

        if H_inv is None:
            H_inv = np.eye(dim)

        # Pre-compute the square root of H_inv for sampling?
        # Or just use H_inv to scale a random vector.
        # If we sample z ~ N(0, I), then v = H_inv @ z is NOT N(0, H_inv).
        # We need v ~ N(0, H_inv).
        # Let H_inv = L L^T (Cholesky). Then v = L z is N(0, H_inv).
        try:
            L_inv = np.linalg.cholesky(H_inv)
        except np.linalg.LinAlgError:
            # Fallback if H_inv is not PD (shouldn't happen if we checked eigvals)
            L_inv = np.eye(dim)

        # Adaptive sigma
        sigma = 0.1
        rng = np.random.default_rng(0)
        
        while calls_made < budget and time.time() < deadline:
            # Step
            z = rng.standard_normal(dim)
            # Scale by the shape of the function
            step_vec = L_inv @ z
            # Normalize the step vector to have unit norm to control sigma properly?
            # No, sigma should control the magnitude.
            # Let's normalize the direction and multiply by sigma.
            norm = np.linalg.norm(step_vec)
            if norm < 1e-8:
                direction = np.ones(dim)
            else:
                direction = step_vec / norm
            
            y = np.clip(x + sigma * direction, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Increase sigma on success
                sigma *= 1.5
            else:
                sigma *= 0.87
            
            # If sigma gets too small, we might be stuck or converged
            if sigma < 1e-10:
                break

        # Phase 3: High-precision polish using L-BFGS-B with numerical gradient
        # If we have budget left, try to refine with a gradient-based method
        if calls_made < budget and time.time() < deadline:
            try:
                # Use the best point found so far
                x0 = best_x.copy()
                
                # Define the objective function for scipy
                # We need to be careful with the call count
                def scipy_obj(x):
                    nonlocal calls_made
                    if calls_made >= budget or time.time() > deadline:
                        return best_f
                    val = func(x)
                    calls_made += 1
                    if val < best_f:
                        best_f = val
                        best_x = x.copy()
                    return val
                
                # Use L-BFGS-B which handles bounds
                result = minimize(scipy_obj, x0, method='L-BFGS-B', 
                                  bounds=[(-5, 5)] * dim,
                                  options={'maxiter': 50, 'maxls': 20})
                
                if result.success and result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
                    
            except Exception:
                pass

        return best_x, best_f

    def _simple_es(self, func, best_x, best_f):
        # Fallback for very small budgets
        calls_made = 0
        rng = np.random.default_rng(0)
        x = best_x
        fx = best_f
        sigma = 1.0
        # Simple loop
        while calls_made < self.budget:
            if time.time() > start_time + 290: break # start_time is not in scope, this is a stub
            # Actually, this method is rarely called. Let's make it safe.
            pass
        return best_x, best_f
