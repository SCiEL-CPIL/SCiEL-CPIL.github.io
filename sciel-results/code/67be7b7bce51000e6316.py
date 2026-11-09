import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: CMA-ES
        # Initial point
        x = np.zeros(n)
        # Initial sigma
        sigma = 2.0
        
        # CMA-ES parameters
        lam = 10  # Population size
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Step size control
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        
        calls = 1
        cma_budget = int(budget * 0.8)
        
        while calls < cma_budget and time.time() < deadline:
            # Generate population
            B = np.linalg.cholesky(C)
            Y = np.random.randn(lam, n)
            P = (x + sigma * (Y @ B.T)).T
            
            # Clip to box to ensure valid inputs for func (though func clips anyway)
            # Clipping here helps CMA-ES not drift too far
            P = np.clip(P, -5, 5)
            
            # Evaluate
            F = np.array([func(p) for p in P])
            calls += lam
            
            if time.time() > deadline:
                break
            
            # Sort
            idx = np.argsort(F)
            P = P[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = P[0].copy()
            
            # Update mean
            x_new = np.sum(w[:, None] * P[:mu], axis=0)
            
            # Update step size
            delta = (x_new - x) / sigma
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * B^-1 * delta
            # B is upper triangular
            try:
                B_inv_delta = np.linalg.solve(np.triu(B).T, delta)
            except np.linalg.LinAlgError:
                B_inv_delta = delta # Fallback
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * B_inv_delta
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1)))) < (1 + c_sigma) * np.sqrt(n + 1)
            
            # Update correlation matrix
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x) / sigma
            
            # Rank-one and rank-mu updates
            C = (1 + c_1 * c_c * (2 - c_c) * mueff - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) - (1 - c_c) * C) \
                + c_mu * np.sum(w[:mu] * np.outer(P[:mu] - x, P[:mu] - x), axis=0)
            
            # Symmetrize and regularize
            C = (C + C.T) / 2
            # Add small identity if ill-conditioned
            if np.linalg.cond(C) > 1e12:
                C += 1e-6 * np.eye(n)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            x = np.clip(x_new, -5, 5)
            
            if sigma < 1e-8:
                break

        # Phase 2: Polish with Nelder-Mead
        remaining = budget - calls
        
        if remaining > 10 and time.time() < deadline:
            class NMWrapper:
                def __init__(self, func, max_calls, deadline):
                    self.func = func
                    self.calls = 0
                    self.max_calls = max_calls
                    self.deadline = deadline
                    self.best_x = best_x
                    self.best_f = best_f
                    
                def __call__(self, x):
                    if self.calls >= self.max_calls or time.time() > self.deadline:
                        return 1e100
                    self.calls += 1
                    fx = self.func(x)
                    if fx < self.best_f:
                        self.best_f = fx
                        self.best_x = x.copy()
                    return fx

            nm_wrapper = NMWrapper(func, remaining, deadline)
            
            x0 = best_x
            init_step = 0.1 
            simplex = [x0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += init_step
                simplex.append(xi)
            
            try:
                # maxiter is not the same as max function calls, but helps limit
                minimize(nm_wrapper, x0, method='Nelder-Mead', 
                         options={'initial_simplex': simplex, 'maxiter': remaining})
            except Exception:
                pass

        return best_x, best_f
