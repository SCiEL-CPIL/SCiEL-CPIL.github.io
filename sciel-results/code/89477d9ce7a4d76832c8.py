import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

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
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                pt = pt.reshape(dim)
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Quadratic Model Construction
        # For Sphere: f(x) = (x-c)^T A (x-c) + f0
        # Expanded: x^T A x - 2 c^T A x + c^T A c + f0
        # Let Q = A, q = -2 A c, const = c^T A c + f0
        # We need dim*(dim+1)/2 + dim + 1 parameters.
        # For dim=5: 15 + 5 + 1 = 21 params.
        
        n_params = dim * (dim + 1) // 2 + dim + 1
        # Use slightly more points for better conditioning
        n_points = n_params + 5  # 26 points
        
        if budget < n_points + 10:
            # Fallback to simple random search if budget is too small
            return self._random_search(func, dim, budget, best_x, best_f, start_time, deadline)

        # Generate points for quadratic fit
        # We want points that span the space well.
        # Center + axis points + random points.
        points = []
        
        # Center
        points.append(np.zeros(dim))
        
        # Axis-aligned points at +/- 3.0 (within [-5,5])
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 3.0
            p2 = np.zeros(dim); p2[i] = -3.0
            points.append(p1)
            points.append(p2)
        
        # Add random points to fill up to n_points
        rng = np.random.default_rng(123)
        while len(points) < n_points:
            p = rng.uniform(-4, 4, dim)
            points.append(p)
        
        points = np.array(points)
        
        # Evaluate all points
        values = np.array([evaluate(p) for p in points])
        
        # Build feature matrix
        # Features: x_i*x_j (i<=j), x_i, 1
        A = np.zeros((n_points, n_params))
        for k, p in enumerate(points):
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    A[k, idx] = p[i] * p[j]
                    idx += 1
            for i in range(dim):
                A[k, idx] = p[i]
                idx += 1
            A[k, idx] = 1.0
        
        # Solve least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_params)
        
        # Extract parameters
        n_quad = dim * (dim + 1) // 2
        Q_flat = coeffs[:n_quad]
        q_vec = coeffs[n_quad:n_quad + dim]
        c_const = coeffs[n_quad + dim]
        
        # Reconstruct Q matrix
        Q = np.zeros((dim, dim))
        idx = 0
        for i in range(dim):
            for j in range(i, dim):
                val = Q_flat[idx]
                Q[i, j] = val
                Q[j, i] = val
                idx += 1
        
        # Find minimizer of quadratic:
        # grad f = 2 Q x + q_vec = 0  =>  x_opt = -0.5 * Q^{-1} q_vec
        try:
            # Check if Q is positive definite enough
            # If not, use pinv
            x_opt = np.linalg.solve(Q, -0.5 * q_vec)
        except np.linalg.LinAlgError:
            x_opt = np.linalg.pinv(Q) @ (-0.5 * q_vec)
        
        x_opt = np.clip(x_opt, -5, 5)
        f_opt_est = evaluate(x_opt)
        
        # Phase 2: Local Polish with CMA-ES or (1+1)-ES
        # Since we have a good starting point, use a local search.
        # (1+1)-ES is simple and effective for smooth functions.
        
        x = best_x.copy()
        fx = best_f
        
        # Initialize sigma based on distance to boundary or previous step
        sigma = 0.5
        
        rng = np.random.default_rng(456)
        
        # Use remaining budget for local search
        # We've used n_points + 1 calls so far.
        
        while calls_made < budget and time.time() < deadline:
            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = x + step
            y = np.clip(y, -5, 5)
            
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Increase sigma slightly on success
                sigma *= 1.2
            else:
                # Decrease sigma on failure
                sigma *= 0.85
            
            # Convergence check
            if sigma < 1e-10:
                break
                
            # If we are very close to the estimated optimum, we can tighten sigma
            dist_to_opt = np.linalg.norm(x - x_opt)
            if dist_to_opt < 1e-6:
                sigma = min(sigma, 1e-6)
                if sigma < 1e-8:
                    break

        return best_x, best_f

    def _random_search(self, func, dim, budget, best_x, best_f, start_time, deadline):
        calls_made = 0
        rng = np.random.default_rng(789)
        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        
        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            pt = np.asarray(pt, dtype=np.float64)
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val
        
        while calls_made < budget and time.time() < deadline:
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = evaluate(y)
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1
            else:
                sigma *= 0.9
            if sigma < 1e-8:
                break
        
        return best_x, best_f
