"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4 and cumulative step-size control."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        
        # Initialize at origin to exploit f14's typical minimum location
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        # CMA-ES parameters
        lambda_ = 4  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.ones(mu)
        weights /= np.sum(weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2 / mu_eff) - 2) / ((dim + 2) ** 2 + mu_eff))
        
        # mu_eff is the inverse effective population size
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Recalculate c_sigma, c_c, c_1, c_mu with correct mu_eff
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2 / mu_eff) - 2) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial sigma
        sigma = 1.2
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for sqrt(C)
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-10))
        Cs = D[:, None] * B.T  # sqrt(C)
        
        # Cumulation paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Chi-squared expectation for step size control
        chi_lambda = np.sqrt(lambda_ * (1 - 1 / (4 * lambda_) - 1 / (21 * lambda_ ** 2)))
        
        for gen in range(200):  # max generations
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * Cs @ z, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((fy, y))
                
            if len(offspring) < mu:
                break
                
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * offspring[i][1]
            x_new = y_w
            
            # Update step size
            d = (y_w - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * d
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_lambda
            h_sigma = min(h_sigma, 1.7 + 2 / (dim + 1))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * d
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 / dim - c_mu / dim) * C \
                + c_1 * (p_c[:, None] * p_c[None, :] - delta_h * C) \
                + c_mu / sigma ** 2 * sum(weights[i] * (offspring[i][1] - x)[:, None] * (offspring[i][1] - x)[None, :] for i in range(mu))
            
            # Ensure positive definiteness
            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-10)
            C = B @ np.diag(D) @ B.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_lambda * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            x = x_new
            fx = func(x) if calls < budget else fx
            if calls < budget:
                calls += 1
            else:
                break
                
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            # Update sqrt(C)
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            Cs = D[:, None] * B.T
            
            # Check for convergence
            if np.linalg.norm(x - best_x) < 1e-6 and sigma < 1e-4:
                break
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
