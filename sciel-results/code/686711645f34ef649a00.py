"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES style optimizer with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Global best tracking
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls_made = 1
        
        # CMA-ES parameters
        lambda_ = 2 * self.dim  # Population size
        c1 = (4 + log(self.dim + 1) - log(self.dim)) / ((self.dim + 4) * (self.dim + 1.3) + 4)
        cmu = min(1 - c1, 2 * (lambda_ - 2 + 1/(self.dim+2)) / ((self.dim + 2) * (lambda_ + 2) - 2))
        cs = (lambda_ + 2 + 2/(self.dim + 1)) / (self.dim + lambda_ + 5 + 2/(self.dim + 1))
        ds = 1 + 2 * max(0, sqrt((cs + 1) / (cs + 1)) - 1) + cs
        chi = sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        # Restart strategy: up to 5 restarts, each with a portion of the budget
        max_restarts = 5
        budget_per_restart = max(50, self.budget // max_restarts)
        
        for restart in range(max_restarts):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Initialize CMA-ES state
            # Start from global best or random if first restart
            if restart == 0:
                x_mean = x_best.copy()
            else:
                # Perturb from global best for restart
                x_mean = np.clip(x_best + rng.normal(0, 1.0, self.dim), -5, 5)
            
            sigma = 2.0  # Initial step size
            C = np.eye(self.dim)
            ps = np.zeros(self.dim)
            pc = np.zeros(self.dim)
            chi = sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
            
            local_best_val = f_best
            local_best_x = x_best.copy()
            local_calls = 0
            
            while calls_made < self.budget and local_calls < budget_per_restart:
                if time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                for i in range(lambda_):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(self.dim)
                    y = x_mean + sigma * (C @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append((y, func(y)))
                    calls_made += 1
                    local_calls += 1
                
                if not offspring:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda x: x[1])
                
                # Update best
                if offspring[0][1] < local_best_val:
                    local_best_val = offspring[0][1]
                    local_best_x = offspring[0][0]
                    if offspring[0][1] < f_best:
                        f_best = offspring[0][1]
                        x_best = offspring[0][0]
                
                # Update mean (weighted)
                mu = lambda_ // 2
                weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1 / np.sum(weights**2)
                
                y_w = np.zeros(self.dim)
                for i in range(mu):
                    y_w += weights[i] * (offspring[i][0] - x_mean)
                x_mean = np.clip(x_mean + y_w, -5, 5)
                
                # Update covariance matrix
                y_w = y_w / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + (1 - (1 - cs)**2) * (C - np.outer(ps, ps) / chi**2)) \
                    + cmu * (np.outer(y_w, y_w) + (1 - (1 - cs)**2) * (C - np.outer(ps, ps) / chi**2))
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update step size control
                ps = (1 - cs) * ps + cs / chi * (y_w / np.linalg.norm(y_w + 1e-10) if np.linalg.norm(y_w) > 1e-10 else 0)
                
                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)
                
                # Check for stagnation
                if sigma < 1e-6 or local_calls > budget_per_restart:
                    break
            
            # If we didn't use full budget for this restart, continue to next

        return x_best, f_best


def log(x):
    return np.log(x)


def sqrt(x):
    return np.sqrt(x)
# EVOLVE-BLOCK-END
