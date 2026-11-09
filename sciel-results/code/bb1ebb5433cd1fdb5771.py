"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start CMA-ES for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # We use a multi-start strategy. 
        # Rastrigin has many local minima. A single CMA-ES might get stuck.
        # We allocate budget to 2-3 runs of CMA-ES.
        # CMA-ES is efficient in 5D.
        
        # Parameters for CMA-ES
        # Population size lambda = 4 + floor(3 log(5)) = 4 + 3*1.6 = 4+4=8? 
        # Standard is 4 + floor(3 log(n)). For n=5, log(5)~1.6, 3*1.6=4.8->4. Lambda=8.
        # Let's use lambda=10 for slightly better exploration.
        lambda_pop = 10
        
        # We will run CMA-ES for a certain number of function evaluations per restart.
        # Total budget 500.
        # Restart 1: 300 evals
        # Restart 2: 200 evals (if time permits and budget remains)
        
        # Since we don't know the time cost of func, we rely on budget and time check.
        
        def run_cmaes(x_start, sigma_start, budget_left):
            nonlocal best_x, best_f, calls_used
            
            if budget_left <= 0:
                return
            
            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(self.dim)
            C_sqrt_inv = np.eye(self.dim)
            
            # CMA-ES parameters
            mu = lambda_pop // 2  # Selection pressure
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Step sizes
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((self.dim + 2)**2 + mu_eff))
            
            # State
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            chi_n = np.sqrt(self.dim) * (1 + 1/(2*self.dim) + 1/(36*self.dim**2))
            
            # Initial evaluation
            # Note: x_start is already evaluated in the main loop? 
            # No, we treat this as a fresh sub-process.
            # But we must be careful not to double count if we reuse points.
            # For simplicity and strict budget control, we evaluate the start point here if not done.
            # However, the main loop manages the global budget.
            
            # Let's structure it so the main loop manages the budget counter.
            # This inner function will just use the budget_left provided.
            
            # Actually, it's cleaner to have a single loop that manages the population.
            # But CMA-ES logic is stateful.
            
            # Let's implement the standard CMA-ES loop here.
            # We need to evaluate the initial mean.
            # To save budget, we can assume the start point was evaluated by the caller?
            # No, let's just evaluate it here and deduct from budget.
            
            if budget_left < 1:
                return

            fx = func(x_mean)
            calls_used += 1
            budget_left -= 1
            if fx < best_f:
                best_f = fx
                best_x = x_mean.copy()
            
            # Loop
            while budget_left >= lambda_pop:
                if time.time() > deadline:
                    return
                
                # Sample population
                # Use Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(self.dim)
                    
                ys = rng.standard_normal((lambda_pop, self.dim))
                # Transform to x
                # x_i = x_mean + sigma * (L @ y_i)^T
                # Efficiently: x_i = x_mean + sigma * (y_i @ L.T)
                xs = x_mean[None, :] + sigma * (ys @ L.T)
                
                # Clip to bounds
                xs = np.clip(xs, -5, 5)
                
                # Evaluate
                fs = np.array([func(x) for x in xs])
                calls_used += lambda_pop
                budget_left -= lambda_pop
                
                # Update best
                idx_min = np.argmin(fs)
                if fs[idx_min] < best_f:
                    best_f = fs[idx_min]
                    best_x = xs[idx_min].copy()
                
                # Sort by fitness
                idx_sorted = np.argsort(fs)
                xs_sorted = xs[idx_sorted]
                fs_sorted = fs[idx_sorted]
                
                # Update mean
                x_new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # Update step size (sigma)
                # p_sigma update
                # y_sigma = C_sqrt_inv @ (x_new_mean - x_mean) / sigma
                # We need C_sqrt_inv. We can compute it from L.
                # C = L L^T. C^{-1/2} is not simply L^{-1}.
                # Standard CMA-ES uses the inverse square root of C.
                # For 5D, we can just compute eigendecomposition or use the fact that 
                # we can maintain C_sqrt_inv.
                # Let's use a simpler update for sigma to avoid complex matrix ops if possible?
                # No, CMA-ES relies on this.
                # Let's compute C^{-1/2} via eigendecomposition of C (symmetric pos def).
                # Or better, use the existing L? 
                # Actually, standard implementation updates C first, then computes C^{-1/2}.
                
                # Let's update C first.
                y_mean = (x_new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * y_mean
                sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                p_c = (1 - c_c) * p_c + c_c * (np.sign(fx - fs_sorted[0]) if False else 1) * y_mean
                # Wait, the sign in p_c update depends on the selection.
                # Standard: p_c = (1-c_c)p_c + c_c * sqrt(mu_eff) * C^{-1/2} (x_new_mean - x_mean)/sigma
                # But we need C^{-1/2}.
                
                # Let's do the matrix updates properly.
                # 1. Compute H = sum w_i (y_i - y_bar) (y_i - y_bar)^T
                # y_i are the normalized steps.
                # y_i = (x_i - x_mean_old) / sigma
                # But we clipped xs, so the steps are distorted. 
                # CMA-ES with clipping is tricky. 
                # Given the small dimension (5) and small budget (500), 
                # and the fact that Rastrigin is smooth enough away from minima,
                # clipping is a minor issue unless we are on the boundary.
                # The minimizer is in [-4, 4], so we likely won't hit [-5, 5] often.
                
                # Let's simplify: 
                # Use the unclipped ys for the covariance update to maintain stability, 
                # or just accept the bias. 
                # Actually, let's just use the standard update on the unclipped values 
                # if we assume clipping is rare. 
                # If clipping is frequent, the covariance will be wrong.
                # Alternative: Use a simpler (mu+lambda)-ES with adaptive step size?
                # The (1+1)-ES scored 1.0. A simple population (mu+lambda)-ES might be better.
                # CMA-ES is overkill for 5D with 500 budget? 
                # CMA-ES converges fast. 
                
                # Let's stick to CMA-ES but handle the inverse sqrt.
                # Compute eigendecomposition of C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Update p_c
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ y_mean)
                
                # Update C
                delta_sig = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) * (np.linalg.norm(p_c) / chi_n - 1)
                # H = sum w_i (y_i - y_bar) (y_i - y_bar)^T
                # y_bar = C^{-1/2} (x_new_mean - x_mean)/sigma = C_inv_sqrt @ y_mean
                y_bar = C_inv_sqrt @ y_mean
                H = np.zeros((self.dim, self.dim))
                for i in range(mu):
                    y_diff = C_inv_sqrt @ ((xs_sorted[i] - x_mean) / sigma) - y_bar
                    H += weights[i] * np.outer(y_diff, y_diff)
                
                C = (1 + c_1 + c_mu - c_1 - c_c * (1 - c_c)) * C + c_1 * (np.outer(p_c, p_c) - C) + c_mu * H
                # Correction: The standard update is:
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-c_c)C) + c_mu H
                # Let's use that.
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * C) + c_mu * H
                
                # Symmetrize
                C = (C + C.T) / 2
                
                x_mean = x_new_mean
            
            return

        # Strategy:
        # 1. Start at center (0,0,0,0,0) or random? 
        # Rastrigin minimum is at 0 (before rotation/shift). 
        # With rotation/shift, the minimum is unknown.
        # Start at 0 is a good bet for Rastrigin structure.
        # Start 1: 0, sigma=2.0
        x_start1 = np.zeros(self.dim)
        run_cmaes(x_start1, 2.0, self.budget // 2)
        
        if calls_used < self.budget and time.time() < deadline:
            # Start 2: Random point
            x_start2 = rng.uniform(-5, 5, self.dim)
            run_cmaes(x_start2, 2.0, self.budget - calls_used)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
