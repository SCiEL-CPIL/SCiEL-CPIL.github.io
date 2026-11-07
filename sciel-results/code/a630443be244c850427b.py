import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES with multi-start strategy for BBOB-style f6.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda = 4 + floor(3 log(dim))
        lambda_pop = int(4 + 3 * np.log(dim))
        # Number of parents mu
        mu = max(1, int(lambda_pop / 2))
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        # Damping for step size control
        d_sigma = c_sigma + 2 * min(1, (mu_eff - 1) / (dim + 1)) * (1 - c_sigma)
        # Chi_lambda for expected norm of N(0, I)
        chi_lambda = np.sqrt(lambda_pop) * (1 - 1/(4*lambda_pop) + 1/(21*lambda_pop**2))
        
        best_x = None
        best_f = float('inf')
        
        # Multi-start strategy: use all budget, but restart if stagnating
        # We will run CMA-ES for chunks of budget.
        # Let's try a single run first, but with a restart mechanism if sigma gets too small.
        # Actually, a simple multi-start with fixed restarts is often more robust for BBOB.
        # Let's split budget into 5 restarts of 100 evaluations each.
        
        num_restarts = 5
        budget_per_restart = budget // num_restarts
        remaining_budget = budget
        
        rng = np.random.default_rng(42)
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            # Start near the center or random? Center is often good for BBOB.
            # Let's start at 0 for the first restart, and random for others.
            if r == 0:
                x_mean = np.zeros(dim)
            else:
                x_mean = rng.uniform(-4, 4, dim)
            
            sigma = 2.0
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            f_mean = func(x_mean)
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()
            
            calls_made = 1
            total_calls_in_restart = budget_per_restart
            if r == num_restarts - 1:
                total_calls_in_restart = remaining_budget
            else:
                remaining_budget -= total_calls_in_restart
                
            # Main loop for this restart
            while calls_made < total_calls_in_restart and time.time() < deadline:
                # Sample lambda points
                # y = B * (D * z)
                # x = x_mean + sigma * y
                Y = []
                for _ in range(lambda_pop):
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    Y.append(y)
                Y = np.array(Y)
                X = x_mean + sigma * Y
                
                # Evaluate
                F = np.array([func(xi) for xi in X])
                calls_made += lambda_pop
                
                # Sort indices
                idx = np.argsort(F)
                F_sorted = F[idx]
                Y_sorted = Y[idx]
                X_sorted = X[idx]
                
                # Update best
                if F_sorted[0] < best_f:
                    best_f = F_sorted[0]
                    best_x = X_sorted[0].copy()
                
                # Update mean
                x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
                
                # Update paths
                y_inv_sigma = (x_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_inv_sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) / chi_lambda < 1.4 + 2 / (dim + 1))
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_inv_sigma
                
                # Update covariance matrix
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
                C += c_1 * np.outer(pc, pc)
                C += c_mu * np.sum(weights[:mu] * np.outer(Y_sorted[:mu], Y_sorted[:mu]), axis=0) / (sigma**2)
                
                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_lambda - 1))
                
                # Update Cholesky factorization (B, D)
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C)
                    D = np.diag(L)
                    B = np.linalg.inv(L.T)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, reset covariance
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    sigma *= 0.5
                    break # Restart this segment? Or just continue with identity?
                    # Let's just reset and continue, but reduce sigma to be safe.
                
                x_mean = x_new
                f_mean = F_sorted[0]
                
                # Check for convergence/stagnation
                if sigma < 1e-6:
                    break
                # If the best point hasn't improved in a long time, we might want to restart early
                # But the fixed restart schedule is simpler and safer.
            
        return best_x, best_f
