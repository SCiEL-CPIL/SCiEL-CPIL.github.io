"""
CMA-ES optimizer for BBOB f3 (Rastrigin).
Uses a robust CMA-ES implementation with a large initial sigma to handle
global exploration and local refinement in a single phase.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES Parameters
        # Population size: 4 + 3 * log(n)
        pop_size = 4 + int(3 * np.log(n))
        if pop_size > budget:
            pop_size = max(2, budget // 2)
        
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        # Initial mean and sigma
        # Start at center, with large sigma to cover the box [-5, 5]^n
        mean = np.zeros(n)
        sigma = 2.0  # Large enough to explore the whole space initially
        
        # Covariance matrix
        C = np.eye(n)
        C_sqrt = np.eye(n)
        
        # Adaptation parameters
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (weights[1] - 1) / ((n + 2)**2 + mu_eff))
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        no_improve = 0
        last_best = float('inf')
        
        # Main Loop
        while not check_deadline():
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(n)
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                update_best(y, fy)
                offspring.append((fy, y))
                
                if check_deadline():
                    break
            
            if len(offspring) < mu:
                break
                
            # Sort offspring
            offspring.sort(key=lambda x: x[0])
            
            # Check improvement
            if offspring[0][0] < last_best - 1e-10:
                last_best = offspring[0][0]
                no_improve = 0
            else:
                no_improve += 1
            
            # Recombination
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * offspring[i][1]
                
            # Update mean
            old_mean = mean.copy()
            mean = y_w
            
            # Update covariance matrix
            ys = (y_w - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * ys
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n < (1.4 + 2 / (n + 1)) * cc)
            if hsig:
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * ys
            else:
                pc = (1 - cs) * pc
            
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                zi = (offspring[i][1] - old_mean) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)
                
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * C) + cmu * rank_mu
            
            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2.0
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_sqrt = np.eye(n)
            
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Convergence criteria
            if sigma < 1e-12:
                break
            if no_improve > 30 and sigma < 0.5:
                # If we are stuck and sigma is small, we might be in a local minimum.
                # For Rastrigin, it's better to just continue until budget is exhausted
                # or restart if we have plenty of budget left.
                # Given the budget is 500 and pop_size ~ 11, we have ~45 generations.
                # Stagnation > 30 gens is very long. Let's just break if we are truly stuck.
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
