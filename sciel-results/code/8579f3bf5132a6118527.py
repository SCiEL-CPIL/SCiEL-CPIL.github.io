import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # 1. Global Phase: CMA-ES
        # 2. Local Phase: Nelder-Mead
        
        # Budget allocation
        # CMA-ES needs at least lambda * 10 generations for convergence?
        # Let's use 300 calls for CMA-ES, 199 for Local.
        
        # Implementing CMA-ES from scratch is complex.
        # Let's use scipy.optimize.differential_evolution if possible, or a simple DE.
        # Actually, for 5D, a well-tuned (mu+lambda)-ES or CMA-ES is best.
        
        # Let's try a simple, robust CMA-ES implementation.
        
        D = self.dim
        # Population size
        lambda_pop = 10 + int(3 * np.log(D))
        # Selection size
        mu = lambda_pop // 2
        
        # Initial state
        x_mean = np.zeros(D)
        sigma = 2.0 # Initial step size
        C = np.eye(D) # Covariance
        pc = np.zeros(D) # Evolution path for C
        ps = np.zeros(D) # Evolution path for sigma
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mu_eff / D) / (D + 4 + 2 * mu_eff / D)
        cs = (mu_eff + 2) / (D + mu_eff + 5)
        c1 = 2 / ((D + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((D + 2)**2 + mu_eff)))
        
        # Budget tracking
        calls = 0
        best_x = x_mean
        best_f = np.inf
        
        # Initial evaluation
        # We start at 0,0,0,0,0? Or random?
        # The minimizer is in [-4, 4]^5. 0 is a good guess.
        x_start = np.zeros(D)
        f_start = func(x_start)
        calls += 1
        best_x = x_start
        best_f = f_start
        
        # CMA-ES Loop
        while calls < self.budget - 100: # Leave 100 for local search
            if time.time() > deadline:
                break
                
            # Sample population
            B = np.linalg.cholesky(C)
            # Standard normal samples
            Z = np.random.randn(lambda_pop, D)
            # Transform
            P = np.dot(Z, B)
            # Apply sigma and mean
            Pop = np.array([x_mean + sigma * p for p in P])
            
            # Evaluate
            Fvals = []
            for x in Pop:
                if time.time() > deadline:
                    break
                # Clip to box [-5, 5]
                x_clipped = np.clip(x, -5, 5)
                f_val = func(x_clipped)
                Fvals.append(f_val)
                calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped
            
            # If we ran out of budget or time in the middle of population, pad
            while len(Fvals) < lambda_pop:
                Fvals.append(np.inf)
                calls += 1 # Account for "wasted" budget or just stop? 
                # Better to just break if we can't complete population
                if calls >= self.budget - 100:
                    break
            
            if len(Fvals) < lambda_pop:
                break
            
            # Sort
            idx = np.argsort(Fvals)
            y = P[idx[:mu]]
            f_sorted = np.array(Fvals)[idx[:mu]]
            
            # Update mean
            x_new = x_mean + sigma * np.dot(w, y)
            
            # Update sigma path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / np.linalg.norm(B) * np.dot(w, y)
            
            # Update C path
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x_mean) / sigma
            
            # Update Covariance
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls // lambda_pop))) < (1.4 + 2 / (D + 1)) * np.sqrt((D + 1) / (D - 1)))
            # Simplified hsig check
            hsig = 1 if np.linalg.norm(ps) < (1.4 + 2 / (D + 1)) * np.sqrt((D + 1) / (D - 1)) else 0
            
            C = (1 + c1 * hsig * (hsig - 1) + cmu) * C - c1 * (pc[:, None] * pc[None, :] + (hsig - 1) * C) - cmu * np.dot(y.T * w, y)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small identity for stability
            C += 1e-10 * np.eye(D)
            
            # Update Sigma
            sigma *= np.exp(cs / (D + 2) * (np.linalg.norm(ps) / np.sqrt((D + 1) / (D - 1)) - 1))
            
            x_mean = x_new
            
            # Check for convergence
            if sigma < 1e-10:
                break
        
        # 2. Local Refinement
        # Use Nelder-Mead from best_x
        if calls < self.budget:
            try:
                res = minimize(func, best_x, method='Nelder-Mead', options={'maxiter': self.budget - calls, 'xatol': 1e-8, 'fatol': 1e-12})
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                pass
        
        return best_x, best_f
