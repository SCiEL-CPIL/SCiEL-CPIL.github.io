import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_fx = np.inf
        rng = np.random.default_rng(42)
        
        # Define restart strategy
        # (start_x, sigma0, max_generations_estimate)
        # We will manage budget dynamically.
        
        # Strategy: 
        # 1. Global search with larger sigma
        # 2. Local search with smaller sigma starting from best
        
        # Let's do 3 phases
        # Phase 1: 100 calls, sigma=2.0, random start
        # Phase 2: 100 calls, sigma=1.0, random start
        # Phase 3: 300 calls, sigma=0.5, start from best of Phase 1&2, then shrink sigma
        
        # Actually, a single CMA-ES run with a large initial sigma and adaptive shrinking might be sufficient and more efficient if it doesn't get stuck.
        # But multi-start is safer for Katsuura.
        
        # Let's implement a helper for CMA-ES
        def run_cmaes(func, x0, sigma0, budget, deadline, rng, best_state):
            n = self.dim
            # Population size lambda
            # lambda = 4 + 3*log(n) is standard, let's use 10 for n=5 to be robust
            lam = 10 
            mu = lam // 2
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1 / np.sum(w**2)
            
            # Adaptation rates
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # State
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_sqrt = np.eye(n) # Cholesky factor
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Precompute constants
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + c_sigma
            dampc = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + c_c
            
            calls_made = 0
            gen = 0
            
            while calls_made < budget and time.time() < deadline:
                # Sample population
                # Generate standard normals
                Z = rng.standard_normal((lam, n))
                # Project to covariance C
                # x_i = x + sigma * Z_i @ C_sqrt
                Y = Z @ C_sqrt
                X = x + sigma * Y
                
                # Evaluate
                F = []
                for i in range(lam):
                    xi = np.clip(X[i], -5, 5)
                    fi = func(xi)
                    F.append((fi, xi))
                    calls_made += 1
                    if time.time() > deadline:
                        break
                
                if not F: break
                if len(F) < lam:
                    # Pad with worst values if we ran out of time/budget mid-generation
                    # This is messy, better to check before starting generation
                    # But we already made calls. We just use what we have.
                    # For simplicity, if we can't complete a generation, we stop.
                    # Actually, we should check calls_made + lam <= budget before starting.
                    # Let's refactor to check at start of loop.
                    pass

                # Sort
                F.sort(key=lambda t: t[0])
                
                # Update best
                if F[0][0] < best_state[0]:
                    best_state[0] = F[0][0]
                    best_state[1] = F[0][1].copy()
                    
                # Update mean
                # x_new = sum(w_i * x_i)
                # Use only top mu
                x_new = np.sum([w[i] * F[i][1] for i in range(mu)], axis=0)
                
                # Update sigma
                # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) / ||C_sqrt|| * (C_sqrt^-1 (x_new - x) / sigma)
                # This requires inverse of C_sqrt.
                # Alternative: update p_sigma in the uncorrelated space?
                # Standard implementation:
                # d = x_new - x
                # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt @ (d / sigma)) / norm(C_sqrt)
                # Wait, the formula is:
                # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) / ||C|| * (C^-1/2 (x_new - x) / sigma)
                # C^-1/2 is inverse of Cholesky factor.
                
                # Let's use a simpler update if C is identity initially, but it changes.
                # We need to maintain C and its Cholesky factor.
                
                # Update C
                # rank-mu update
                # delta_sig = (x_new - x) / sigma
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - (1 - c_sigma)^2) chi_n^2 I) + c_mu sum(w_i (y_i y_i^T))
                # where y_i = (x_i - x) / sigma
                
                # This is getting complex to implement from scratch correctly and fast.
                # Is there a simpler, robust optimizer?
                # SciPy's `differential_evolution` or `basinhopping`?
                # `differential_evolution` is robust but might be slow for 500 calls in 5D.
                # `basinhopping` uses local minimizer (L-BFGS-B or Nelder-Mead).
                
                # Let's stick to a simplified CMA-ES or use `scipy.optimize.minimize` with Nelder-Mead for local polishing after a global search.
                
                # Global Search: Random restarts with Nelder-Mead?
                # Nelder-Mead is fast per call.
                # Strategy:
                # 1. Generate 50 random points. (50 calls)
                # 2. Pick best 10.
                # 3. Run Nelder-Mead from each of the 10 best, budget 40 calls each. (400 calls)
                # 4. Total 450 calls.
                # Nelder-Mead is often very effective for Katsuura if started in a good basin.
                
                break # Exit CMA-ES implementation attempt, switch strategy
                
            return best_state

        # New Strategy: Multi-Start Nelder-Mead
        # Nelder-Mead is simple and fast.
        # We need to implement it or use scipy.
        # Using scipy.optimize.minimize is allowed.
        
        from scipy.optimize import minimize
        
        # Phase 1: Global Exploration
        # Use 20% of budget for random sampling
        n_explore = total_budget // 5
        explore_x = []
        explore_f = []
        
        for i in range(n_explore):
            if time.time() > deadline: break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            explore_x.append(x)
            explore_f.append(fx)
            
        # Sort and take top 5
        indices = np.argsort(explore_f)[:5]
        
        best_fx = np.inf
        best_x = None
        
        # Phase 2: Local Polishing
        # Use remaining budget for 5 Nelder-Mead runs
        remaining_budget = total_budget - len(explore_f)
        budget_per_polish = remaining_budget // 5
        
        for idx in indices:
            if time.time() > deadline: break
            if budget_per_polish <= 0: break
            
            x0 = explore_x[idx]
            f0 = explore_f[idx]
            
            # Nelder-Mead options
            # maxiter is not exactly function calls, but we can limit it.
            # Nelder-Mead in scipy doesn't have a strict function call limit in options, 
            # but we can wrap the function to count.
            
            call_count = [0]
            def func_wrapper(x):
                call_count[0] += 1
                return func(x)
                
            # We need to ensure we don't exceed budget_per_polish
            # scipy Nelder-Mead might make many calls.
            # We can use a callback to stop? No, minimize doesn't support callback stopping cleanly in all versions.
            # Better: Implement a simple Nelder-Mead or use a fixed maxiter and hope.
            # Or use `basinhopping`?
            
            # Let's try a simple custom Nelder-Mead or just use `minimize` with `method='Nelder-Mead'` and `options={'maxiter': 200}`.
            # 200 iterations * ~5 function calls/iter = 1000 calls. Too many.
            # We need to limit function calls.
            
            # Let's implement a simple (1+1)-ES or Nelder-Mead with strict call counting.
            # Actually, the previous (1+1)-ES was okay but maybe not optimal.
            
            # Let's try a different approach: **Covariance Matrix Adaptation (CMA-ES)** is the gold standard here.
            # I will implement a robust, compact CMA-ES.
            
            pass

        return best_x, best_fx
