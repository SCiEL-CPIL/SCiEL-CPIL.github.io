import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def f(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return 1e10
            calls += 1
            xc = np.clip(x, -5.0, 5.0)
            val = float(func(xc))
            if val < best_f:
                best_f = val
                best_x = xc.copy()
            return val

        # 1. Initial exploration to find a good starting point for CMA-ES
        # This helps with early progress in AOCC scoring.
        rng_init = np.random.default_rng(42)
        
        # Evaluate center and a few random points
        candidates = []
        x_center = np.zeros(dim)
        f_center = f(x_center)
        candidates.append((f_center, x_center.copy()))
        
        for _ in range(5):
            if time.time() > deadline or calls >= budget:
                break
            x_try = rng_init.uniform(-4.0, 4.0, dim)
            fx_try = f(x_try)
            candidates.append((fx_try, x_try.copy()))
        
        candidates.sort(key=lambda t: t[0])
        x_start = candidates[0][1].copy()
        f_start = candidates[0][0]

        # 2. CMA-ES Optimization
        # Parameters
        popsize = 4 + int(3 * np.log(dim))  # 8 for dim=5
        sigma = 1.5  # Initial step size
        c1 = (4 - dim + 1 / dim) / ((dim + 4) ** 2 + 4)
        cmu = 2 * (2 - c1) ** 2 - (c1 - 2 / (dim + 1)) ** 2
        cs = (dim + 1.7) / (dim + 9 + mu_eff) if False else (dim + 1.7) / (dim + 9 + (2 * (2 - c1) ** 2 - (c1 - 2 / (dim + 1)) ** 2) / 2) # Simplified cs
        # Standard cs
        mu_eff = 1 / c1 - 1
        cs = (dim + 1.7) / (dim + 9 + mu_eff)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # CMA-ES State
        m = x_start.copy()  # Mean
        C = np.eye(dim)     # Covariance matrix
        C_inv = np.eye(dim) # Inverse covariance
        C_sqrt = np.eye(dim) # Square root of C
        C_inv_sqrt = np.eye(dim) # Inverse square root of C
        
        # Weighted recombination
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update counters
        gen = 0
        max_iter = 100  # Safety cap on generations
        
        while calls < budget and time.time() < deadline and gen < max_iter:
            gen += 1
            
            # Sample points
            ys = []
            xs = []
            fs = []
            
            for i in range(popsize):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng_init.standard_normal(dim)
                y = C_sqrt @ z
                x = m + sigma * y
                fx = f(x)
                ys.append(y)
                xs.append(x)
                fs.append(fx)
            
            if len(ys) == 0:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update mean
            for i in range(mu):
                m += weights[i] * (xs[i] - m)
                
            # Update evolution paths
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys[i]
                
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_w
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt @ y_w)
            
            # Check for divergence
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) > (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(1 / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                
            # Update covariance matrix
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
            h_sig = 1.0 if h_sig else 0.0
            
            C = (1 + c1 * h_sig * (np.outer(p_c, p_c) - C) + c1 + cmu) * C
            # Wait, the update rule is:
            # C <- (1 - c1 - cmu + c1*h_sig*(||pc||^2 - (1-(1-cs)^2)*tr(C))) * C + c1 * pc*pc^T + cmu * sum(wi * (yi - yw)*(yi - yw)^T)
            
            # Let's use a simpler, more stable implementation from scratch to avoid subtle bugs
            # I will implement a simplified CMA-ES that is known to be robust.
            
            # Actually, let's restart with a cleaner CMA-ES implementation.
            break

        # Re-implementing a clean, robust CMA-ES
        # The above loop broke out. Let's do it properly in a dedicated function or block.
        
        # Since the previous loop was messy, I will replace the entire optimization logic with a clean CMA-ES.
        
        # Reset state for clean CMA-ES
        # We already used some budget for initial exploration.
        
        # Clean CMA-ES Implementation
        m = x_start.copy()
        sigma = 1.5
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Precompute weights
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        c1 = (4 - dim + 1 / dim) / ((dim + 4) ** 2 + 4)
        cmu = 2 * (2 - c1) ** 2 - (c1 - 2 / (dim + 1)) ** 2
        cs = (dim + 1.7) / (dim + 9 + mu_eff)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky factorization helper
        def cholesky_inv_sqrt(C):
            L = np.linalg.cholesky(C)
            L_inv = np.linalg.inv(L)
            C_inv_sqrt = L_inv.T
            return L, C_inv_sqrt, L_inv.T @ L_inv # C_inv
        
        L, C_inv_sqrt, C_inv = cholesky_inv_sqrt(C)
        
        while calls < budget and time.time() < deadline:
            # Sample
            ys = []
            xs = []
            fs = []
            
            for i in range(popsize):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng_init.standard_normal(dim)
                y = L @ z
                x = m + sigma * y
                fx = f(x)
                ys.append(y)
                xs.append(x)
                fs.append(fx)
            
            if len(ys) == 0:
                break
                
            # Sort
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update mean
            for i in range(mu):
                m += weights[i] * (xs[i] - m)
                
            # Weighted recombination vector
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys[i]
                
            # Update evolution paths
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_w
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ y_w)
            
            # Update sigma
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) > (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(1 / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                
            # Update covariance
            h_sig = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            
            # C update
            C = (1 - c1 - cmu + c1 * h_sig * (np.linalg.norm(p_c) ** 2 - (1 - (1 - cs) ** 2) * dim)) * C
            C += c1 * np.outer(p_c, p_c)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(ys[i] - y_w, ys[i] - y_w)
                
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky
            try:
                L, C_inv_sqrt, C_inv = cholesky_inv_sqrt(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add small jitter
                C += 1e-8 * np.eye(dim)
                L, C_inv_sqrt, C_inv = cholesky_inv_sqrt(C)
                
            gen += 1
            if gen > 100: # Safety break
                break

        return best_x, best_f
