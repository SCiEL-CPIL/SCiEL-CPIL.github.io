import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f, deadline
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls_made >= self.budget:
                raise ValueError("Budget exceeded")
            
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != self.dim:
                raise ValueError("Wrong dimension")
            if not np.all(np.isfinite(x)):
                raise ValueError("Non-finite input")
                
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        if best_x is None:
             # Initial point at origin (center of box)
             x0 = np.zeros(self.dim)
             try:
                 wrapped_func(x0)
             except:
                 return np.zeros(self.dim), np.inf

        # CMA-ES Implementation
        n = self.dim
        # Population size: 4 + 3*ceil(3*log(n)) is standard, but for n=5, lambda=6 is good
        lam = 6
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        cs = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(c1 / 2, (2 * (mueff - 2 + 1 / mueff)) / ((n + 2)**2 + mueff))
        # Damping for step size control
        ds = 1 + 2 * max(0, sqrt(mueff/n - 1) - 1) + cs
        
        # Initial values
        mean = best_x.copy()
        sigma = 2.5  # Large initial step to cover the box
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # History of best values for stagnation detection
        best_f_hist = []
        
        try:
            while calls_made < self.budget and time.time() < deadline - 1.0:
                # Generate population
                pop = []
                for i in range(lam):
                    # Sample from N(mean, sigma^2 * C)
                    # Use Cholesky decomposition for efficiency
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # If C is not positive definite, reset it
                        C = np.eye(n)
                        L = np.eye(n)
                    
                    z = rng.standard_normal(n)
                    x_new = mean + sigma * (L @ z)
                    pop.append(x_new)
                
                # Evaluate
                pop_f = []
                for x in pop:
                    try:
                        fx = wrapped_func(x)
                        pop_f.append(fx)
                    except (TimeoutError, ValueError):
                        break
                
                if len(pop_f) < lam:
                    break

                # Sort
                sort_idx = np.argsort(pop_f)
                pop_sorted = [pop[i] for i in sort_idx]
                pop_f_sorted = [pop_f[i] for i in sort_idx]
                
                # Update best
                if pop_f_sorted[0] < best_f:
                    best_f = pop_f_sorted[0]
                    best_x = pop_sorted[0].copy()
                
                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += w[i] * pop_sorted[i]
                
                # Update covariance matrix
                # ps update
                inv_sqrt_C = np.linalg.inv(np.sqrtm(C))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ (new_mean - mean) / sigma)
                
                # pc update
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * chi_n)
                pc = (1 - cc) * pc + hsig * cc * (new_mean - mean) / sigma
                
                # C update
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C
                for i in range(mu):
                    yi = (pop_sorted[i] - mean) / sigma
                    C += c1 * np.outer(yi, yi)
                    C += cmu * w[i] * np.outer(yi, yi)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update sigma
                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                mean = new_mean
                
                # Stagnation check: if sigma becomes very small or C collapses, restart
                if sigma < 1e-6 or np.linalg.cond(C) > 1e12:
                    # Restart with a smaller sigma around the best point
                    mean = best_x.copy()
                    sigma = 0.5
                    C = np.eye(n)
                    pc = np.zeros(n)
                    ps = np.zeros(n)
        
        except (TimeoutError, ValueError):
            pass
        except Exception:
            pass

        return best_x, best_f
