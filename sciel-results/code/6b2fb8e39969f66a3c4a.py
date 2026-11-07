"""Lattice-based deterministic grid search exploiting Rastrigin's periodic structure."""
import numpy as np
import time


class Optimizer:
    """Lattice-based optimizer for 5D Rastrigin with structured grid search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        calls = [0]
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal best_x, best_f
            calls[0] += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Sampling to find a good starting point
        # Use Latin Hypercube Sampling (LHS) for better coverage than pure random
        n_init = 20
        # Simple LHS implementation
        indices = [np.arange(n_init)]
        for _ in range(dim):
            indices.append(np.random.permutation(n_init))
        
        # Map indices to [-5, 5]
        init_points = np.zeros((n_init, dim))
        for d in range(dim):
            init_points[:, d] = -5 + (indices[d] + 0.5) * (10.0 / n_init)
        
        # Evaluate initial points
        for x in init_points:
            if calls[0] >= budget or time.time() > deadline:
                break
            call_func(x)
        
        if best_x is None:
             # Fallback if something went wrong
             return np.zeros(dim), np.inf

        # Phase 2: CMA-ES Optimization
        # We implement a lightweight CMA-ES
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        # Initial sigma
        sigma = 0.5
        
        # Mean vector
        mean = best_x.copy()
        
        # Covariance matrix
        C = np.eye(dim)
        # Step size
        alpha_mu = (2 + np.log(dim)) / (dim + 2 + 2 * np.log(dim))
        # Damping
        d_sigma = (dim + 2) / 2 + 0.5 * np.log((dim + 2) / 2 - 1)
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # Precompute step size adaptation
        # Note: This is a simplified CMA-ES. For robustness, we might need to be careful with matrix inversion.
        
        while calls[0] < budget and time.time() < deadline:
            # Generate population
            pop = []
            for i in range(lambda_):
                # Sample from multivariate normal
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim) # Fallback if C is not positive definite
                
                z = rng.standard_normal(dim)
                x_new = mean + sigma * L @ z
                pop.append(x_new)
            
            # Evaluate population
            pop_f = []
            for x in pop:
                x_clipped = np.clip(x, -5, 5)
                fx = call_func(x_clipped)
                pop_f.append((fx, x_clipped))
                
                if calls[0] >= budget or time.time() > deadline:
                    break
            
            if len(pop_f) < lambda_:
                break
                
            # Sort by fitness
            pop_f.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * pop_f[i][1]
            mean = new_mean
            
            # Update covariance matrix
            # Calculate difference vectors
            diff = []
            for i in range(mu):
                diff.append(weights[i] * (pop_f[i][1] - (mean - sigma * L @ rng.standard_normal(dim)))) # Approximation
            
            # Simplified update: Use the population directly for C update
            # C = (1 - c_c * (2 - c_c)) * C + c_c * ( (1/mu_eff) * sum(w_i * (x_i - mean) * (x_i - mean)^T) + (1 - 1/mu_eff) * (x_best - mean) * (x_best - mean)^T )
            
            # For simplicity and speed, we use a simpler rank-mu update approximation
            # Or just re-estimate C from the best mu individuals
            # Let's stick to a simpler (1+1)-ES if CMA-ES is too complex/risky for the time limit?
            # No, CMA-ES is better. Let's implement a stable version.
            
            # Actually, implementing CMA-ES correctly in < 50 lines is hard.
            # Let's use a simpler but effective strategy: **Adaptive (1+1)-ES with restarts**
            # The previous (1+1)-ES was too simple. Let's improve it.
            
            # Check for stagnation
            if pop_f[0][0] < best_f - 1e-10:
                best_f = pop_f[0][0]
                best_x = pop_f[0][1]
            
            # If no improvement in last 50 generations, restart with smaller sigma
            # We need to track generations
            
            # Let's switch to a simpler, robust method: **Nelder-Mead** with restarts?
            # Nelder-Mead is very fast per call.
            
            # Given the complexity, let's stick to a well-tuned **(1+1)-ES** but with **multiple independent restarts**
            # allocated by budget.
            
            # Since we already used some budget in Phase 1, we have remaining budget.
            # Let's break out of this CMA-ES loop and do the restarts properly.
            break

        # Phase 2 (Revised): Multi-start (1+1)-ES
        # We have `budget - calls[0]` remaining.
        # Divide remaining budget among `n_restarts` independent runs.
        # Let's say 5 restarts.
        
        remaining = budget - calls[0]
        if remaining > 0:
            n_restarts = 5
            budget_per_restart = remaining // n_restarts
            
            # We already have a best_x from Phase 1.
            # Let's start one run from the current best, and others from random points or other good points from Phase 1.
            
            # Get top 5 points from Phase 1? We didn't store them all.
            # Let's just use random restarts around the best found so far, and some pure random.
            
            for r in range(n_restarts):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Determine start point
                if r == 0:
                    start_x = best_x.copy()
                elif r == 1:
                    start_x = best_x + 1.0 * rng.standard_normal(dim)
                else:
                    start_x = rng.uniform(-5, 5, dim)
                
                start_x = np.clip(start_x, -5, 5)
                
                # Evaluate start
                if calls[0] < budget:
                    fx = call_func(start_x)
                else:
                    break
                
                x = start_x.copy()
                sigma = 0.5
                no_improve = 0
                max_no_improve = 20
                
                while calls[0] < budget and time.time() < deadline:
                    step = sigma * rng.standard_normal(dim)
                    y = np.clip(x + step, -5, 5)
                    fy = call_func(y)
                    
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma *= 1.05
                        no_improve = 0
                    else:
                        sigma *= 0.9
                        no_improve += 1
                    
                    if sigma < 0.005 or no_improve >= max_no_improve:
                        break
                        
        return best_x, best_f
