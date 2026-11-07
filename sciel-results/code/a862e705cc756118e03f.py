import numpy as np
import time

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

        # Phase 1: Sphere-specific Linear Fit
        # Model: f(x) = a * ||x||^2 + w^T x + c
        # This is linear in parameters (a, w, c).
        # For a standard sphere f(x) = ||x - center||^2 + offset, a=1.
        # center = -w / (2a)
        
        # Generate points: Origin + +/- R * e_i
        R = 2.0
        points = [np.zeros(dim)]
        for i in range(dim):
            p_pos = np.zeros(dim); p_pos[i] = R
            p_neg = np.zeros(dim); p_neg[i] = -R
            points.append(p_pos)
            points.append(p_neg)
        
        n_points = len(points)
        if budget < n_points + 5:
             # Fallback to simple ES if budget is tiny
             return self._simple_es(func, best_x, best_f)

        # Evaluate points
        values = np.array([evaluate(p) for p in points])
        
        # Construct design matrix
        # Columns: ||x||^2, x_0, ..., x_{dim-1}, 1
        A = np.zeros((n_points, dim + 2))
        for i, p in enumerate(points):
            A[i, 0] = np.dot(p, p)
            A[i, 1:dim+1] = p
            A[i, dim+1] = 1.0
            
        b = values
        
        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            a = coeffs[0]
            w = coeffs[1:dim+1]
            c_const = coeffs[dim+1]
            
            # Check if the quadratic term is significant and positive
            if a > 1e-6:
                # Center estimate
                x_center = -w / (2.0 * a)
                x_center = np.clip(x_center, -5, 5)
                f_center = evaluate(x_center)
                
                if f_center < best_f:
                    best_x = x_center
                    best_f = f_center
                    
        except Exception:
            pass

        # Phase 2: Polish with (1+1)-ES
        # Start from best_x
        x = best_x.copy()
        fx = best_f
        
        # Adaptive sigma
        # Start small since we are likely close
        sigma = 0.1
        rng = np.random.default_rng(0)
        
        # Track success for sigma adaptation
        success_rate = 0.5
        window_size = 10
        successes = []
        
        while calls_made < budget and time.time() < deadline:
            # Generate random direction
            z = rng.standard_normal(dim)
            
            # Step
            y = np.clip(x + sigma * z, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                successes.append(1)
                # Increase sigma
                sigma *= 1.2
            else:
                successes.append(0)
                # Decrease sigma
                sigma *= 0.9
                
            # Adapt sigma based on recent success rate
            if len(successes) >= window_size:
                recent_success_rate = np.mean(successes[-window_size:])
                # If success rate is too high, increase sigma more
                # If too low, decrease more
                if recent_success_rate > 0.6:
                    sigma *= 1.1
                elif recent_success_rate < 0.3:
                    sigma *= 0.9
                successes = [] # Reset window
                
            # If sigma gets too small, we might be stuck or converged
            if sigma < 1e-10:
                break

        return best_x, best_f

    def _simple_es(self, func, best_x, best_f):
        # Fallback for very small budgets
        calls_made = 0
        rng = np.random.default_rng(0)
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        
        start_time = time.time()
        deadline = start_time + 290
        
        while calls_made < self.budget and time.time() < deadline:
            z = rng.standard_normal(self.dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.2
            else:
                sigma *= 0.9
                
        return x, fx
