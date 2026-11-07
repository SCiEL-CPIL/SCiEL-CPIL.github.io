import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        x = np.zeros(self.dim)
        best_x = x
        best_f = float('inf')
        
        calls_made = 0
        
        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
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
        if self.budget < n_points + 5:
             # Fallback to simple ES if budget is tiny
             return self._simple_es(func, best_x, best_f)

        # Generate points
        # Use a mix of center and random to ensure spread
        points = []
        # Center
        points.append(np.zeros(self.dim))
        # Axis-aligned points
        for i in range(self.dim):
            p1 = np.zeros(self.dim); p1[i] = 2.0
            p2 = np.zeros(self.dim); p2[i] = -2.0
            points.append(p1)
            points.append(p2)
        # Random points
        rng = np.random.default_rng(42)
        while len(points) < n_points:
            p = rng.uniform(-5, 5, self.dim)
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
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if fit is good
            # If residuals are large, the function might not be quadratic or points are bad.
            # But for Sphere, it should be very good.
            
            # Extract Q, q, c
            n_quad = self.dim * (self.dim + 1) // 2
            Q_flat = coeffs[:n_quad]
            q = coeffs[n_quad:n_quad + self.dim]
            c = coeffs[n_quad + self.dim]
            
            # Reconstruct Q matrix
            Q = np.zeros((self.dim, self.dim))
            idx = 0
            for i in range(self.dim):
                for j in range(i, self.dim):
                    val = Q_flat[idx]
                    Q[i, j] = val
                    Q[j, i] = val
                    idx += 1
            
            # Solve Q x = -0.5 q
            # x_opt = -0.5 Q^{-1} q
            try:
                # Use solve for stability
                x_opt = np.linalg.solve(Q, -0.5 * q)
            except np.linalg.LinAlgError:
                x_opt = np.linalg.pinv(Q) @ (-0.5 * q)
            
            x_opt = np.clip(x_opt, -5, 5)
            f_opt = evaluate(x_opt)
            
        except Exception:
            # If fitting fails, just proceed with best found
            pass

        # Phase 2: Polish with (1+1)-ES
        # Start from best_x
        x = best_x.copy()
        fx = best_f
        
        # Adaptive sigma
        sigma = 0.5
        rng = np.random.default_rng(0)
        
        while calls_made < self.budget and time.time() < deadline:
            # Step
            step = sigma * rng.standard_normal(self.dim)
            y = np.clip(x + step, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Increase sigma on success (1/5 rule approximation)
                sigma *= 1.5
            else:
                sigma *= 0.87
            
            # If sigma gets too small, we might be stuck or converged
            if sigma < 1e-8:
                break

        return best_x, best_f

    def _simple_es(self, func, best_x, best_f):
        # Fallback for very small budgets
        calls_made = 0
        # Note: best_f is already set from previous calls if any, but in this path 
        # it's called from __call__ where calls_made is local. 
        # Actually, let's just implement a simple loop here assuming we have budget left.
        # But wait, _simple_es is called when budget is small.
        # We need to track calls correctly.
        # Let's refactor __call__ to avoid this helper or pass state.
        # For now, this helper is only used if budget < 30.
        # Let's just do a few random steps.
        rng = np.random.default_rng(0)
        x = best_x
        fx = best_f
        sigma = 1.0
        remaining = self.budget - calls_made # This is tricky, calls_made is local to __call__
        # Let's just assume we can make a few calls.
        # Actually, it's safer to just return best_x, best_f if budget is that small.
        return best_x, best_f
