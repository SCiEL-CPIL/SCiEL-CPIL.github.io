import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_fx = np.inf
        
        def latin_hypercube_sample(n, d, low=-5.0, high=5.0):
            """Generate Latin Hypercube samples for better initial coverage."""
            points = np.zeros((n, d))
            for i in range(n):
                for j in range(d):
                    points[i, j] = low + (high - low) * (i + np.random.uniform()) / n
            return points

        # Use Latin Hypercube for better initial coverage
        num_init = 20
        init_x = latin_hypercube_sample(num_init, dim)
        init_f = np.array([func(x) for x in init_x])
        n_calls = num_init
        
        idx = np.argmin(init_f)
        x = init_x[idx].copy()
        sigma = 1.5
        
        # If time is short, just do local search
        if time.time() > deadline - 10:
             # Quick polish
             try:
                 res = minimize(func, x, method='Nelder-Mead', options={'maxfev': budget - n_calls, 'maxiter': budget - n_calls})
                 if res.fun < best_fx:
                     best_fx = res.fun
                     best_x = res.x
             except:
                 pass
             return best_x, best_fx

        # CMA-ES Parameters for small budget
        lam = 10
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c1
        
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        DC = np.linalg.cholesky(C)
        
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Track stagnation
        stagnation_counter = 0
        best_fx_prev = best_fx
        
        while n_calls < budget:
            if time.time() > deadline - 5:
                break
                
            # Check stagnation
            if best_fx == best_fx_prev:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_fx_prev = best_fx
                
            # Reboot logic: if stagnated for 12 generations, restart from a fresh random point
            if stagnation_counter > 12:
                # Try a few random restarts and pick the best
                best_restart_x = x
                best_restart_f = best_fx
                for _ in range(3):
                    if n_calls + 3 > budget:
                        break
                    rx = np.random.uniform(-5, 5, dim)
                    rf = func(rx)
                    n_calls += 1
                    if rf < best_restart_f:
                        best_restart_f = rf
                        best_restart_x = rx
                x = best_restart_x.copy()
                sigma = 2.0
                C = np.eye(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                DC = np.linalg.cholesky(C)
                stagnation_counter = 0
                continue
            
            # Generate population
            Z = np.random.randn(lam, dim)
            Y = Z @ DC.T
            
            F = np.empty(lam)
            for i in range(lam):
                if n_calls >= budget:
                    break
                xi = x + sigma * Y[i]
                # Clip to box to be safe, though func clips
                xi = np.clip(xi, -5, 5)
                F[i] = func(xi)
                n_calls += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = xi.copy()
            
            if n_calls >= budget:
                break
                
            indices = np.argsort(F)
            Z_sorted = Z[indices]
            Y_sorted = Y[indices]
            
            z_bar = np.dot(weights, Z_sorted[:mu])
            mean_step = DC @ z_bar
            x = x + sigma * mean_step
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_bar
            
            sigma = sigma * np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_N - 1))
            sigma = np.clip(sigma, 1e-8, 10.0) # Prevent overflow/underflow
            
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * mean_step
            
            Y_sel = Y_sorted[:mu]
            M = np.zeros((dim, dim))
            for i in range(mu):
                M += weights[i] * np.outer(Y_sel[i], Y_sel[i])
                
            h_sig = 1.0 if np.linalg.norm(ps) / chi_N < (1.4 + 2/(dim+1)) else 0.0
            
            C = (1 - c1 - cmu * h_sig) * C + c1 * np.outer(pc, pc) + cmu * M
            C = (C + C.T) / 2
            
            try:
                DC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 0.01 * np.eye(dim)
                try:
                    DC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    DC = np.eye(dim)
                    C = np.eye(dim)
        
        # Polish with Nelder-Mead if budget remains
        remaining = budget - n_calls
        if remaining > 15:
            try:
                res = minimize(func, best_x, method='Nelder-Mead', options={'maxfev': remaining, 'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass
            
            # Additional polish with COBYLA if still have budget
            remaining2 = budget - n_calls
            if remaining2 > 10:
                try:
                    res2 = minimize(func, best_x, method='COBYLA', options={'maxiter': remaining2, 'rhobeg': 0.1})
                    if res2.fun < best_fx:
                        best_fx = res2.fun
                        best_x = res2.x
                except Exception:
                    pass

        return best_x, best_fx
