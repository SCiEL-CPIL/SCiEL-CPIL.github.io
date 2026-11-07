"""Adaptive (mu+lambda)-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280

        best_x = None
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        def es_run(x0, sigma0, max_iter=100):
            nonlocal total_calls, best_f, best_x
            n = dim
            lam = 20
            mu = lam // 2
            weights = np.ones(mu) / mu
            
            mean = x0.copy()
            sigma = sigma0
            best_local_x = mean.copy()
            best_local_f = np.inf
            
            # Initial population
            points = []
            fvals = []
            for i in range(lam):
                if total_calls >= budget or time.time() > deadline:
                    return best_local_x, best_local_f
                z = rng.standard_normal(n)
                x = np.clip(mean + sigma * z, -5, 5)
                fx = safe_call(x)
                points.append(x)
                fvals.append(fx)
                if fx < best_local_f:
                    best_local_f = fx
                    best_local_x = x.copy()
            
            for it in range(max_iter):
                if total_calls >= budget or time.time() > deadline:
                    break
                
                order = np.argsort(fvals)
                points = [points[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if fvals[0] < best_local_f:
                    best_local_f = fvals[0]
                    best_local_x = points[0].copy()
                
                # Update mean
                mean = np.sum([weights[i] * points[i] for i in range(mu)], axis=0)
                
                # Simple sigma adaptation based on spread
                spread = np.mean([np.linalg.norm(points[i] - mean) for i in range(mu)])
                if spread > 0.1:
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    
                sigma = np.clip(sigma, 1e-6, 50.0)
                
                # Generate next population
                points = []
                fvals = []
                for i in range(lam):
                    if total_calls >= budget or time.time() > deadline:
                        return best_local_x, best_local_f
                    z = rng.standard_normal(n)
                    x = np.clip(mean + sigma * z, -5, 5)
                    fx = safe_call(x)
                    points.append(x)
                    fvals.append(fx)
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local_x = x.copy()
                        
            return best_local_x, best_local_f

        # Initial start
        x0 = rng.uniform(-5, 5, dim)
        f0 = safe_call(x0)
        
        x_best, f_best = es_run(x0, 2.0, max_iter=100)
        
        # Restart loop
        while total_calls < budget and time.time() < deadline:
            # Perturb best point
            x_restart = x_best + rng.standard_normal(dim) * 0.5
            x_restart = np.clip(x_restart, -5, 5)
            # Use a moderate sigma for restart
            x_best, f_best = es_run(x_restart, 1.0, max_iter=50)
            
            # If no improvement, we might want to break, but let's keep going to use budget
            # unless we are stuck.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
