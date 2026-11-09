"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Multi-Start CMA-ES with Adaptive Restarts.
    
    The Bent Cigar function (BBOB f12) is highly ill-conditioned and "bent", 
    meaning it has a narrow valley that curves. A single CMA-ES run might 
    get stuck exploring the wrong part of the valley or stagnate if the 
    covariance matrix adapts poorly to the bend.
    
    By using multiple short restarts, we:
    1. Increase the chance of finding the "right" entry point to the valley.
    2. Avoid long-term stagnation in a single run.
    3. Allow the covariance matrix to reset and re-adapt to the local geometry 
       around the current best point if the previous run failed to converge.
    
    We implement a simplified CMA-ES (using eigen-decomposition for sampling) 
    and run it for a fixed number of generations per restart, updating the 
    global best solution across all restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        self.deadline = self.start_time + 290.0
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            if self.calls > self.budget:
                raise ValueError("Budget exceeded")
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # CMA-ES Parameters
        N = 10  # Population size
        mu = N // 2
        
        # Weights: log-uniform
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = (mu_eff - 2 + 1 / c_1) / ((dim + 2)**2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)
        
        chi_N = np.sqrt(dim * (1 - 1/(4*dim) + 1/(21*dim**2)))

        # Strategy: Multi-Start
        # We will run multiple independent CMA-ES instances.
        # Each instance gets a portion of the budget.
        # To maximize early progress (AOCC), we should do many short runs initially,
        # then perhaps longer runs if we find a good spot?
        # Actually, for AOCC, every call counts. 
        # A good strategy is to run several "exploratory" runs with small budgets,
        # and then one "exploitative" run with the remaining budget starting from the best found.
        
        # Let's define a schedule:
        # 1. 3 initial runs with budget 50 each (150 calls total).
        # 2. 1 main run with the remaining budget (350 calls).
        
        # However, if the first runs are bad, we waste budget.
        # Alternative: 
        # Run 1: Budget 100.
        # If improvement is slow, restart.
        # Simple heuristic: Fixed number of restarts.
        
        # Let's try:
        # Restart 1: Budget 100
        # Restart 2: Budget 100
        # Restart 3: Budget 100
        # Restart 4 (Main): Remaining budget (~200)
        
        # Actually, let's be more dynamic.
        # We will run CMA-ES for a fixed number of generations, say G.
        # If we haven't improved significantly, we restart.
        # But "significant" is hard to define without knowing f_opt.
        
        # Let's stick to a fixed schedule for robustness:
        # 4 restarts.
        # Budgets: 80, 80, 80, 260.
        # This ensures we have some exploration at the start.
        
        budgets = [80, 80, 80, budget - 240]
        # Ensure last budget is positive
        if budgets[-1] < 10:
             # Fallback to equal distribution
             n_restarts = 5
             b_per = budget // n_restarts
             budgets = [b_per] * n_restarts
             budgets[-1] = budget - b_per * (n_restarts - 1)

        for b_re in budgets:
            if self.calls >= budget:
                break
            if time.time() > self.deadline - 5.0: # Leave 5s for cleanup
                break
            
            # Initialize CMA-ES state
            # Start from a random point in the box, or near the best found so far?
            # For multi-start, random is better for exploration.
            # But if we have a good best_x, starting near it might be better for exploitation.
            # Let's start the first restart at 0, and subsequent ones randomly, 
            # or near the best with some noise.
            
            if self.best_x is None:
                x_mean = np.zeros(dim)
            else:
                # Start near the best found, but with some perturbation to escape local minima
                # Perturbation size: 1.0 seems reasonable for dim 5 in [-5,5]
                x_mean = self.best_x + np.random.randn(dim) * 1.0
                x_mean = np.clip(x_mean, -5, 5)
            
            sigma = 2.0
            C = np.eye(dim)
            C_eigenvalues = np.ones(dim)
            C_eigenvectors = np.eye(dim)
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Local best tracking for this restart
            local_best_f = self.best_f
            local_best_x = self.best_x.copy() if self.best_x is not None else np.zeros(dim)
            
            max_gen = b_re // N
            if max_gen < 2:
                continue
                
            for gen in range(max_gen):
                if time.time() > self.deadline - 5.0:
                    break
                if self.calls + N > self.budget:
                    break
                    
                # 1. Sample N points
                samples = []
                for i in range(N):
                    z = np.random.randn(dim)
                    sqrt_eigs = np.sqrt(C_eigenvalues)
                    y = C_eigenvectors @ (sqrt_eigs * z)
                    x_i = x_mean + sigma * y
                    samples.append(x_i)
                
                # 2. Evaluate
                f_values = []
                for x_i in samples:
                    if time.time() > self.deadline - 5.0:
                        break
                    if self.calls >= self.budget:
                        break
                    f_val = wrapped_func(x_i)
                    f_values.append(f_val)
                    
                if len(f_values) < N:
                    break
                    
                # 3. Sort by fitness
                sorted_indices = np.argsort(f_values)
                samples = [samples[i] for i in sorted_indices]
                f_values = [f_values[i] for i in sorted_indices]
                
                # Update local best
                if f_values[0] < local_best_f:
                    local_best_f = f_values[0]
                    local_best_x = samples[0].copy()
                
                # 4. Update mean
                x_mean_new = np.zeros(dim)
                for i in range(mu):
                    x_mean_new += weights[i] * samples[i]
                
                # 5. Update p_sigma and sigma
                diff = x_mean_new - x_mean
                inv_sqrt_eigs = 1.0 / np.sqrt(C_eigenvalues)
                C_inv_sqrt_diff = C_eigenvectors @ (inv_sqrt_eigs * (C_eigenvectors.T @ diff))
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_inv_sqrt_diff
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_N - 1))
                
                # 6. Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
                
                threshold = (1 + 2.0/c_sigma) * chi_N**2
                denom = 1 - (1 - c_sigma)**(2 * (gen + 1))
                if denom > 1e-10:
                    ratio = (norm_p_sigma**2) / denom
                    h_sigma = 1.0 if ratio < threshold else 0.0
                else:
                    h_sigma = 1.0
                
                sum_yyT = np.zeros((dim, dim))
                for i in range(mu):
                    diff_i = samples[i] - x_mean
                    y_i = C_eigenvectors @ (inv_sqrt_eigs * (C_eigenvectors.T @ diff_i))
                    sum_yyT += weights[i] * np.outer(y_i, y_i)
                
                C_new = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum_yyT
                C_new = (C_new + C_new.T) / 2.0
                
                C_eigenvalues, C_eigenvectors = np.linalg.eigh(C_new)
                C_eigenvalues = np.clip(C_eigenvalues, 1e-10, None)
                
                C = C_new
                x_mean = x_mean_new
                
                if sigma < 1e-8:
                    break

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
