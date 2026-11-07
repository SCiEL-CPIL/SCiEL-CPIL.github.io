import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 285  # 15s margin before 300s kill
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if self.calls >= budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            self.calls += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if np.isfinite(f) and f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def local_optimize(x0, max_iters=100, tol=1e-8):
            """Run L-BFGS-B from x0."""
            if self.calls >= budget - 10 or time.time() > self.deadline - 5:
                return x0, wrapped_func(x0)
            
            # L-BFGS-B requires a function that returns (f, g) if jac=True, 
            # but here we only have f. So we use finite differences (default).
            # We limit the number of iterations and function evaluations.
            # maxfun is the maximum number of function evaluations.
            # We allocate a portion of the remaining budget.
            remaining = budget - self.calls
            maxfun = max(10, min(remaining // 2, 100))
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxfun': maxfun, 'maxiter': max_iters, 'ftol': tol, 'gtol': tol}
                )
                if res.success or res.fun < self.best_f:
                    return res.x, res.fun
            except Exception:
                pass
            return x0, wrapped_func(x0)

        # Phase 1: Generate diverse starting points
        np.random.seed(123)
        starts = []
        starts.append(np.zeros(dim))
        
        # Random points in the box
        for _ in range(10):
            starts.append(np.random.uniform(-4, 4, dim))
        
        # Points near corners
        for corner in range(2 ** dim):
            if self.calls >= budget - 50:
                break
            point = np.array([(corner >> i) & 1 for i in range(dim)], dtype=float)
            point = point * 4 - 2  # Map to [-2, 2]
            starts.append(point)
        
        # Limit starts to avoid using too much budget
        max_starts = min(len(starts), 15)
        starts = starts[:max_starts]
        
        # Phase 2: For each start, run L-BFGS-B
        for i, x0 in enumerate(starts):
            if self.calls >= budget - 20:
                break
            if time.time() > self.deadline - 15:
                break
            
            x0 = np.clip(x0, -5, 5)
            
            # Evaluate starting point
            f0 = wrapped_func(x0)
            if not np.isfinite(f0):
                continue
            
            # Run L-BFGS-B
            try:
                x_opt, f_opt = local_optimize(x0, max_iters=50, tol=1e-6)
                
                if np.isfinite(f_opt) and f_opt < self.best_f:
                    self.best_f = f_opt
                    self.best_x = np.clip(x_opt, -5, 5).copy()
            except Exception:
                pass
        
        # Phase 3: Polish the best point with high-precision L-BFGS-B
        if self.calls < budget - 10 and time.time() < self.deadline - 10:
            try:
                x_polished, f_polished = local_optimize(self.best_x, max_iters=100, tol=1e-12)
                
                if np.isfinite(f_polished) and f_polished < self.best_f:
                    self.best_f = f_polished
                    self.best_x = np.clip(x_polished, -5, 5).copy()
            except Exception:
                pass
        
        # Phase 4: If budget remains, do more restarts around the best point
        if self.calls < budget - 15 and time.time() < self.deadline - 15:
            for _ in range(8):
                if self.calls >= budget - 10 or time.time() > self.deadline - 10:
                    break
                
                # Perturb best point
                noise_scale = 0.2 * np.exp(-self.calls / 100)
                noise = np.random.randn(dim) * noise_scale
                x_pert = np.clip(self.best_x + noise, -5, 5)
                f_pert = wrapped_func(x_pert)
                
                if np.isfinite(f_pert) and f_pert < self.best_f:
                    self.best_f = f_pert
                    self.best_x = x_pert.copy()
                
                # Quick L-BFGS-B polish
                try:
                    x_opt, f_opt = local_optimize(self.best_x, max_iters=20, tol=1e-8)
                    if np.isfinite(f_opt) and f_opt < self.best_f:
                        self.best_f = f_opt
                        self.best_x = np.clip(x_opt, -5, 5).copy()
                except Exception:
                    pass
        
        return self.best_x, self.best_f
