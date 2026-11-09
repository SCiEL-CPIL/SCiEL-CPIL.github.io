import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 300.0  # Safety margin before 330s kill

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 12
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2.0 / (n**2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (n**2 + 2))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Initialization
        # Start at the center of the box, which is a good prior for BBOB
        mean = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        B = np.linalg.cholesky(C)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Pre-compute standard normals? No, generate each time.
        rng = np.random.default_rng(42)
        
        best_x = mean
        best_f = float('inf')
        calls_made = 0
        
        # Initial population evaluation to set best
        # Actually, CMA-ES generates a population each step.
        # We need to ensure we don't exceed budget.
        
        while calls_made < budget:
            if time.time() > self.deadline:
                break
                
            # Generate population
            z = rng.standard_normal((lambda_, n))
            # Transform to y
            y = mean + sigma * (z @ B.T)
            
            # Clip to box
            y = np.clip(y, -5.0, 5.0)
            
            # Evaluate
            f_vals = np.empty(lambda_)
            for i in range(lambda_):
                f_vals[i] = func(y[i])
                calls_made += 1
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_x = y[i].copy()
                
                # Check budget mid-evaluation
                if calls_made >= budget:
                    break
                    # If we break here, f_vals is partially filled. 
                    # We need to handle this carefully. 
                    # It's better to check before the loop if we have enough budget for the whole pop.
            
            if calls_made >= budget:
                break
                
            # Sort indices
            idx = np.argsort(f_vals)
            
            # Update mean
            new_mean = np.sum(weights[:, None] * y[idx[:mu]], axis=0)
            
            # Update covariance matrix and sigma
            # Standard CMA-ES update
            # v = (new_mean - mean) / sigma
            v = (new_mean - mean) / sigma
            
            # Update ps
            ps = (1 - c_1) * ps + np.sqrt(c_1 * (2 - c_1) * mu_eff) / chi_n * v
            
            # Update pc
            hsig = (np.sqrt((1 - (1 - c_1)**20) - 1) < 1.4 + 2 / (n + 1)) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_1)**(2 * (calls_made // lambda_))) / chi_n < 1.4 + 2 / (n + 1))
            
            if hsig:
                pc = (1 - c_c) * pc + c_c * v
            else:
                pc = (1 - c_c) * pc
                
            # Delta h
            delta_h = (hsig - 1) * c_c * (1 - c_c)
            
            # Update C
            # C = (1 - c_1 - c_mu + delta_h) * C + c_1 * pc * pc.T + c_mu * sum(w_i * y_i * y_i.T)
            # Note: y_i here are the standardized steps z_i, or the actual steps?
            # In standard CMA-ES, we update based on the selected steps in the standardized space or transformed space?
            # The standard update uses the selected points y_i (transformed).
            # Let's use the standard formula:
            # C = (1 - c_1 - c_mu + delta_h) * C + c_1 * pc * pc.T + c_mu * sum_{i=1}^mu w_i * (y_i - mean_old) (y_i - mean_old).T / sigma^2 ?
            # Actually, the update is on the covariance of the search distribution.
            # The steps are z_i. The transformed steps are y_i - mean_old = sigma * B z_i.
            # The covariance of the steps is sigma^2 C.
            # The update rule is:
            # C = (1 - c_1 - c_mu + delta_h) * C + c_1 * pc * pc.T + c_mu * sum w_i * (y_i - mean_old)/sigma * ((y_i - mean_old)/sigma).T
            
            steps = (y[idx[:mu]] - mean) / sigma
            C = (1 - c_1 - c_mu + delta_h) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum(weights[:, None, None] * steps[:, :, None] * steps[:, None, :], axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Cholesky decomposition
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, regularize
                C = C + 1e-6 * np.eye(n)
                B = np.linalg.cholesky(C)
                
            # Update sigma
            sigma = sigma * np.exp(c_1 / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Safety: keep mean in box? CMA-ES mean can drift.
            # If mean goes out of box, the clipped points will be on the boundary.
            # This is usually fine, but can stall.
            # We do not clip the mean itself in standard CMA-ES.
            
        return best_x, best_f
