"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        rng = np.random.default_rng(0)
        dim = self.dim
        
        # CMA-ES Parameters
        lambda_pop = 9
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)
        
        cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
        cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
        c1 = 2 / ((lambda_pop + 2) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2) ** 2 + mu_eff))
        
        # State
        x_mean = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls = 1
        
        def update_state():
            nonlocal C, B, D
            # Eigen-decomposition
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(D2)
            # Ensure positive definite
            D = np.maximum(D, 1e-4)
            
        while calls < self.budget:
            if time.time() >= deadline:
                break
            
            # Sample lambda points
            offspring = []
            for i in range(lambda_pop):
                # z ~ N(0, I)
                z = rng.standard_normal(dim)
                # y = B * D * z
                y = B @ (D * z)
                # x = x_mean + sigma * y
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                offspring.append((x_new, func(x_new)))
                calls += 1
                if calls >= self.budget:
                    break
            
            if calls >= self.budget:
                break
                
            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            
            # Update mean
            x_new_mean = np.sum([weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0)
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (B.T @ (x_new_mean - x_mean))))
            
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_pop + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(lambda_pop + 1 - mu_eff) * np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_pop + 1))):
                sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1))
            
            # Update covariance
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_pop + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(lambda_pop + 1 - mu_eff))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((dim, dim))
            for i in range(min(mu, len(offspring))):
                y_i = (offspring[i][0] - x_mean) / sigma
                rankmu += weights[i] * np.outer(y_i, y_i)
                
            C = (1 + c1 * hsig * (cc - cc - 1) + c1 + cmu) * C + c1 * (cc - cc - 1 + hsig * (1 + cc) - 1) * rank1 + cmu * rankmu
            
            # Simplified update often used to avoid numerical issues in manual implementations:
            # C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            # Let's use the simplified stable version
            C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new_mean
            
            # Update best
            best_offspring = offspring[0]
            if best_offspring[1] < best_f:
                best_f = best_offspring[1]
                best_x = best_offspring[0].copy()
                
            # Update basis
            update_state()
            
            # Check convergence (optional, but good)
            if np.linalg.norm(D) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
