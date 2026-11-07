# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        # Number of parents
        mu = lambda_ // 2
        # Weights for the parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance reduction factor
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 2.0  # Initial step size, covers a good portion of [-5, 5]
        
        # Mean
        x_mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        # Inverse square root of C for sampling
        B, D = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-20))
        C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = None
        best_f = np.inf
        n_calls = 0
        
        def evaluate(x):
            nonlocal best_x, best_f, n_calls
            x = np.asarray(x, dtype=np.float64).clip(-5, 5)
            f = float(func(x))
            n_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation of the mean (optional, but helps centering)
        # We skip evaluating the mean directly to save budget for exploration, 
        # or we can just start the loop. Let's start the loop.
        
        # We will run generations until budget is exhausted or time runs out
        while n_calls < budget:
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break

            # Sample lambda_ points
            ys = []
            xs = []
            for _ in range(lambda_):
                # Standard normal
                z = np.random.randn(n)
                # Transform to CMA distribution
                y = C_inv_sqrt @ z
                x = x_mean + sigma * y
                xs.append(x)
                ys.append(y)
            
            # Evaluate
            fvals = np.array([evaluate(x) for x in xs])
            
            # Sort by fitness
            order = np.argsort(fvals)
            xs = np.array(xs)[order]
            fvals = fvals[order]
            ys = np.array(ys)[order]
            
            # Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
            
            # Update p_sigma
            # h_sigma: Heaviside function to check if p_sigma is consistent with p_c
            # Simplified: check if ||p_sigma|| is within expected range
            norm_p_sigma = np.linalg.norm(p_sigma)
            E_norm = np.sqrt(mu_eff) * (1 - np.exp(-0.5 * c_sigma))
            # More standard check:
            # E[||N(0,1)||] approx sqrt(n)
            # Here we use the standard CMA-ES check
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 2/(n+1)) * np.sqrt((1 - (1-c_sigma)**(2*(budget - n_calls)))) * np.sqrt(mu_eff) else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ x_new_mean / sigma - C_inv_sqrt @ x_mean / sigma)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            # Clip sigma to avoid explosion or collapse
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Update covariance C
            # Rank-1 update
            v = (x_new_mean - x_mean) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (v @ v.T + (1 - h_sigma) * c_c * (2 - c_c) * p_c @ p_c.T / sigma**2) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * (xs[:mu] - x_mean)[:, np.newaxis, :] @ (xs[:mu] - x_mean)[:, np.newaxis, :].T / sigma**2, axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update inverse square root
            # Use SVD for stability
            U, s, Vt = np.linalg.svd(C)
            # Ensure positive eigenvalues
            s = np.maximum(s, 1e-12)
            C_inv_sqrt = (Vt.T @ np.diag(1.0 / np.sqrt(s)) @ U.T)
            
            x_mean = x_new_mean
            
            # Stagnation check
            if best_f - fvals[0] < 1e-12 and sigma < 1e-4:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
