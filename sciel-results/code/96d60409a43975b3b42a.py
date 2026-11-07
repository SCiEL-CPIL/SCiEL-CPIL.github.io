"""CMA-ES based optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation with budget and time constraints."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        # CMA-ES parameters
        n = self.dim
        # Initial population size
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        
        # Initial mean: center of the box
        x_mean = np.zeros(n)
        
        # Initial sigma: large enough to explore the [-5, 5]^n box
        # Standard CMA-ES heuristic for sigma_0
        sigma_0 = 2.0 
        
        # Weights and weights for recombination
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step-size adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Hessian matrix update parameters
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # State variables
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Initial evaluation
        x = x_mean.copy()
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls = 1
        
        # Main loop
        while calls < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            xs = []
            fs = []
            
            # Pre-compute B and D for Cholesky decomposition of C
            # To save time, we can use a simpler approach or cache if possible, 
            # but standard CMA-ES requires updating C each step.
            # For n=5, Cholesky is very fast.
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C
                C = np.eye(n)
                B = np.eye(n)
            
            for i in range(lambda_):
                z = np.random.standard_normal(n)
                y = x_mean + sigma_0 * (B @ z)
                y = np.clip(y, -5, 5)
                xs.append(y)
                
                # Check time inside loop if budget is large, but 500 is small
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break
                
                f_y = func(y)
                fs.append(f_y)
                calls += 1
                
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()
            
            if not xs:
                break
            
            # Sort by fitness
            sorted_indices = np.argsort(fs)
            xs_sorted = np.array([xs[i] for i in sorted_indices])
            fs_sorted = np.array([fs[i] for i in sorted_indices])
            
            # Update mean
            x_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # Update sigma
            ys_sorted = (xs_sorted[:mu] - x_mean) / sigma_0
            norm_ys = np.linalg.norm(ys_sorted, axis=1)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ np.mean(ys_sorted, axis=0))
            
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lambda_ + 1))) < (1.4 + 1 / n) * chi_n:
                sigma_0 *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            v = np.mean(ys_sorted, axis=0)
            pc = (1 - c_c) * pc + c_c * (v - x_mean) # Note: v is in y-space, x_mean update used in CMA
            # Actually, standard CMA-ES update for pc uses the weighted mean of y
            # Let's stick to standard formulas carefully
            
            # Correction: Standard CMA-ES updates
            # 1. Generate x_i = m + sigma * B z_i
            # 2. Sort
            # 3. m_new = sum w_i x_(i)
            # 4. y_i = (x_i - m_old) / sigma
            # 5. ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * (B^T * mean(y_(i)))
            # 6. sigma *= exp(c_sigma/d_sigma * (||ps||/chi_n - 1))
            # 7. pc = (1-c_c)pc + c_c * (mean(y_(i)) - m_old) -- Wait, pc is in y-space usually?
            # No, pc is in x-space in some implementations, y-space in others.
            # Let's use the formulation where C is updated in y-space (normalized) or x-space.
            # Common formulation:
            # C = (1 - c_1 - c_mu)C + c_1 (pc pc^T + (1 - (1-c_c)^2) C) + c_mu sum w_i y_(i) y_(i)^T
            # Where y_(i) are the sorted y-vectors.
            
            # Let's re-calculate pc correctly in y-space
            pc = (1 - c_c) * pc + c_c * (np.mean(ys_sorted, axis=0))
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * C)
            C += c_mu * np.sum(weights[:, None, None] * np.outer(ys_sorted[:mu], ys_sorted[:mu]), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # If we didn't use all lambda_ due to time/budget, break
            if len(fs) < lambda_:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
