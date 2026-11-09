from scipy.optimize import minimize
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Sobol sampling to find a good starting point
        # Use 20 points
        n_init = 20
        # Simple Sobol-like using Halton or just random? 
        # Let's use a simple deterministic grid + random mix for robustness
        # Actually, let's just use random with a good seed for simplicity and speed
        rng = np.random.default_rng(42)
        
        for i in range(n_init):
            x = rng.uniform(-5.0, 5.0, n)
            try_func(x)
            
        # Phase 2: CMA-ES
        # Parameters
        lambda_ = 8 # Population size
        mu = lambda_ // 2 # Number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial state
        mean = best_x.copy()
        sigma = 2.0 # Initial step size
        
        B = np.eye(n)
        D = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Main loop
        while calls < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                # Sample z ~ N(0, I)
                z = rng.standard_normal(n)
                # Transform
                x = mean + sigma * (B @ (D @ z))
                # Clip to bounds
                x = np.clip(x, -5.0, 5.0)
                f = try_func(x)
                offspring.append((f, x))
                
            if calls >= budget:
                break
                
            # Sort offspring
            offspring.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.sum([weights[i] * offspring[i][1] for i in range(mu)], axis=0)
            
            # Update evolution paths
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D @ (new_mean - mean)))
            # pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mu_eff) * (new_mean - mean) / sigma
            
            # Simplified update for stability
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (new_mean - mean) / sigma
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (new_mean - mean) / sigma
            
            # Update covariance matrix
            # H = ps * ps.T + (||ps||^2 / chi_n^2 - 1) * pc * pc.T
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (||pc||^2 / n - 1) * I) + cmu * sum(weights * (xi - mean)(xi - mean).T)
            
            # To save time and complexity, we can use a simpler adaptation or just rely on the mean update and sigma adaptation.
            # A full CMA-ES is complex. Let's use a simpler "Adaptive Step Size" method.
            
            # Alternative: Just use Nelder-Mead or L-BFGS-B from the best point found so far?
            # The current program uses L-BFGS-B. 
            # Let's try a different approach: 
            # 1. Find best point.
            # 2. Use Nelder-Mead with adaptive simplex size.
            
            # Let's abort CMA-ES implementation if it's too long to write correctly.
            # Instead, let's use a robust "Ridge Finding" heuristic.
            
            # Break out of CMA-ES loop and do Ridge Search
            break

        # Phase 3: Ridge Search
        # Estimate ridge direction from the best point
        x_best = best_x
        f_best = best_f
        
        # Try to find the direction of least change
        # Sample 10 random directions
        best_dir = None
        best_var = np.inf
        eps = 0.1
        
        for _ in range(10):
            d = rng.standard_normal(n)
            d /= np.linalg.norm(d)
            
            # Evaluate at x_best +/- eps*d
            f_plus = try_func(x_best + eps * d)
            f_minus = try_func(x_best - eps * d)
            
            # Variance is a proxy for curvature
            var = (f_plus - f_minus) ** 2
            if var < best_var:
                best_var = var
                best_dir = d
                
        if best_dir is None:
            return best_x, best_f

        # Now search along this direction
        # Use Golden Section Search
        a = -5.0
        b = 5.0
        # Find a good interval first
        # Expand from x_best
        step = 0.1
        t_left = 0.0
        t_right = 0.0
        f_left = f_best
        f_right = f_best
        
        # Expand left
        while t_left > -5.0 and calls < budget:
            t_left -= step
            x_left = x_best + t_left * best_dir
            x_left = np.clip(x_left, -5.0, 5.0)
            f_left = try_func(x_left)
            if f_left < f_best:
                f_best = f_left
                best_x = x_left
            else:
                break
            step *= 1.5
            
        # Expand right
        step = 0.1
        while t_right < 5.0 and calls < budget:
            t_right += step
            x_right = x_best + t_right * best_dir
            x_right = np.clip(x_right, -5.0, 5.0)
            f_right = try_func(x_right)
            if f_right < f_best:
                f_best = f_right
                best_x = x_right
            else:
                break
            step *= 1.5

        # If we found a better point, refine with Golden Section
        if calls < budget:
            # Define function of t
            def f_t(t):
                x = x_best + t * best_dir
                x = np.clip(x, -5.0, 5.0)
                return try_func(x)
                
            # Golden section search in [t_left, t_right]
            gr = (np.sqrt(5) - 1) / 2
            c = t_right - gr * (t_right - t_left)
            d = t_left + gr * (t_right - t_left)
            fc = f_t(c)
            fd = f_t(d)
            
            while (t_right - t_left) > 1e-4 and calls < budget:
                if fc < fd:
                    t_right = d
                    d = c
                    fd = fc
                    c = t_right - gr * (t_right - t_left)
                    fc = f_t(c)
                else:
                    t_left = c
                    c = d
                    fc = fd
                    d = t_left + gr * (t_right - t_left)
                    fd = f_t(d)
                    
            # Final evaluation
            t_mid = (t_left + t_right) / 2
            x_mid = x_best + t_mid * best_dir
            x_mid = np.clip(x_mid, -5.0, 5.0)
            f_mid = try_func(x_mid)
            if f_mid < best_f:
                best_f = f_mid
                best_x = x_mid

        # Phase 4: Final Polish with L-BFGS-B
        if calls < budget - 10 and time.time() < deadline:
            def obj(x):
                return try_func(x)
            
            try:
                res = minimize(obj, best_x, method='L-BFGS-B', 
                              bounds=[(-5, 5)] * n,
                              options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
