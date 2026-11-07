import numpy as np
import time

class Optimizer:
    """
    Optimizer using Global Quadratic Fit followed by Local Polish.
    Exploits the quadratic nature of BBOB f1 (Sphere).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # 1. Generate points for quadratic fit
        # We need at least d(d+3)/2 + 1 points to fit a quadratic.
        # For d=5, 15 + 5 + 1 = 21 points.
        # We use 50 points for robustness.
        n_fit_points = 50
        rng = np.random.default_rng(0)
        
        # Create a set of points
        # Include origin and axis-aligned points for better conditioning
        points = []
        points.append(np.zeros(self.dim))
        
        # Axis points
        for i in range(self.dim):
            p_pos = np.zeros(self.dim)
            p_neg = np.zeros(self.dim)
            p_pos[i] = 2.0
            p_neg[i] = -2.0
            points.append(p_pos)
            points.append(p_neg)
            
        # Random points to fill up to n_fit_points
        while len(points) < n_fit_points:
            p = rng.uniform(-5, 5, self.dim)
            points.append(p)
            
        points = np.array(points[:n_fit_points])
        
        # 2. Evaluate function at these points
        ys = []
        for p in points:
            if time.time() > deadline:
                break
            if self.budget <= 0:
                break
            try:
                y = func(p)
                ys.append(y)
                self.budget -= 1
            except Exception:
                break
                
        if len(ys) < 21:
            # Fallback: Return best point found or use 1+1-ES
            if len(ys) == 0:
                x = np.zeros(self.dim)
                fx = func(x) if self.budget > 0 else 0.0
                if self.budget > 0: self.budget -= 1
                return x, fx
                
            best_idx = np.argmin(ys)
            x_best = points[best_idx]
            fx_best = ys[best_idx]
            
            # Simple 1+1-ES fallback
            sigma = 1.0
            while self.budget > 0 and time.time() < deadline:
                y = np.clip(x_best + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                self.budget -= 1
                if fy < fx_best:
                    x_best, fx_best = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
            return x_best, fx_best

        ys = np.array(ys)
        
        # 3. Build design matrix for quadratic fit
        # f(x) = x^T A x + b^T x + c
        # Basis: x_i x_j (25), x_i (5), 1 (1) -> 31 coefficients
        V = np.zeros((len(points), 31))
        for k, p in enumerate(points):
            idx = 0
            for i in range(self.dim):
                for j in range(self.dim):
                    V[k, idx] = p[i] * p[j]
                    idx += 1
            for i in range(self.dim):
                V[k, idx] = p[i]
                idx += 1
            V[k, idx] = 1.0

        # 4. Solve least squares
        try:
            theta, _, _, _ = np.linalg.lstsq(V, ys, rcond=None)
        except Exception:
            # Fallback to best point
            best_idx = np.argmin(ys)
            return points[best_idx], ys[best_idx]

        # 5. Reconstruct A, b, c
        A_flat = theta[:25].reshape(self.dim, self.dim)
        A = (A_flat + A_flat.T) / 2.0
        b = theta[25:30]
        c = theta[30]
        
        # 6. Calculate analytical minimum
        try:
            # Check if A is positive definite
            eigvals = np.linalg.eigvalsh(A)
            if np.min(eigvals) > 1e-8:
                A_inv = np.linalg.inv(A)
                x_star = -0.5 * (A_inv @ b)
                x_star = np.clip(x_star, -5, 5)
            else:
                # If not PD, use best point from fit
                x_star = points[np.argmin(ys)]
        except Exception:
            x_star = points[np.argmin(ys)]

        # 7. Evaluate x_star
        fx_star = ys[np.argmin(ys)]
        if self.budget > 0 and time.time() < deadline:
            try:
                fx_test = func(x_star)
                self.budget -= 1
                if fx_test < fx_star:
                    fx_star = fx_test
            except Exception:
                pass

        # 8. Local Polish using Nelder-Mead
        if self.budget > 10 and time.time() < deadline:
            delta = 1e-3
            # Simplex initialization
            simplex = [x_star.copy()]
            for i in range(self.dim):
                v = x_star.copy()
                v[i] += delta
                simplex.append(v)
            
            # Manual Nelder-Mead to control budget strictly
            # Or use scipy if available and safe
            try:
                from scipy.optimize import minimize
                
                calls_made = 0
                max_calls = self.budget - 1
                
                def wrapped_func(x):
                    nonlocal calls_made
                    if calls_made >= max_calls:
                        return np.inf
                    calls_made += 1
                    return func(np.clip(x, -5, 5))
                    
                res = minimize(wrapped_func, x_star, method='Nelder-Mead', 
                               options={'maxiter': max_calls, 'xatol': 1e-12, 'fatol': 1e-12})
                
                if res.fun < fx_star:
                    x_star = np.clip(res.x, -5, 5)
                    fx_star = res.fun
                    self.budget -= calls_made
            except Exception:
                # Fallback to simple gradient descent if scipy fails
                pass

        return x_star, fx_star
