"""Rewritten program using robust CMA-ES with IPOP restarts for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = np.zeros(d)
        best_f = np.inf
        
        # IPOP-CMA-ES parameters
        # Start with a population size that allows good exploration
        lambda_ = 10
        min_lambda = 4
        max_restarts = 100
        restart_count = 0
        
        # Initial sigma: The box is [-5, 5]. A sigma of 2.0 is reasonable.
        # Lunacek global min is often around -3 to -4 in each dim.
        initial_sigma = 2.0
        
        # Stagnation detection
        stagnation_limit = 20 * d # Generations without improvement
        
        while restart_count < max_restarts and budget > 0:
            if time.time() - start_time > timeout:
                break
                
            # Initialize CMA-ES state
            # For the first restart, start at 0. For subsequent, start near best.
            if restart_count == 0:
                m = np.zeros(d)
                sigma = initial_sigma
            else:
                # Perturb around best found
                m = best_x + np.random.randn(d) * initial_sigma * 0.1
                # Reset sigma to a larger value to explore new regions
                sigma = initial_sigma * (1.0 + 0.5 * np.random.rand())

            # CMA-ES hyperparameters
            lambda_ = max(min_lambda, int(4 + 3 * np.log(d) + restart_count * 0.5))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2) ** 2 + mu_eff))
            
            pc = np.zeros(d)
            ps = np.zeros(d)
            C = np.eye(d)
            B = np.eye(d)
            D = np.ones(d)
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (32 * d * d))
            
            # Evaluate initial mean
            f_m = func(m)
            budget -= 1
            if f_m < best_f:
                best_f = f_m
                best_x = m.copy()
            
            last_improvement_gen = 0
            gen = 0
            
            while budget >= lambda_ and time.time() - start_time < timeout:
                gen += 1
                
                # Generate population
                X = []
                for i in range(lambda_):
                    z = np.random.randn(d)
                    y = m + sigma * B @ (D * z)
                    # Clip to box
                    y = np.clip(y, -5, 5)
                    X.append(y)
                
                # Evaluate population
                F = np.array([func(x) for x in X])
                budget -= lambda_
                
                # Update global best
                idx_best = np.argmin(F)
                if F[idx_best] < best_f:
                    best_f = F[idx_best]
                    best_x = X[idx_best].copy()
                    last_improvement_gen = gen
                
                # Sort
                idx = np.argsort(F)
                x_sorted = [X[i] for i in idx]
                
                # Update mean
                m_new = np.zeros(d)
                for i in range(mu):
                    m_new += weights[i] * x_sorted[i]
                
                # Update step size control variable ps
                # Standard CMA-ES: ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * B^{-1}(m_new-m)/sigma
                # Since we have B and D, B^{-1} is B.T. And (m_new-m)/sigma is the step.
                # Note: The step in the standard algorithm is y = (m_new - m) / sigma.
                # But we generated x = m + sigma * B @ (D * z).
                # So the "natural" step is B @ (D * z).
                # The update for ps is usually in the "B-space".
                # Let's stick to the standard implementation found in the CMA-ES reference.
                
                # Calculate the step in the transformed space
                # y = (m_new - m) / sigma
                y = (m_new - m) / sigma
                # Transform y back to the "z" space? No, ps is in the B-space.
                # ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * (B.T @ y)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y)
                
                # Update sigma
                # sigma = sigma * exp(c_sigma / chi_d * (||ps|| - chi_d))
                hsig = np.linalg.norm(ps) / chi_d
                if hsig < (1.4 + 2 / (d + 1)):
                    sigma *= np.exp(c_sigma / chi_d * (hsig - 1))
                
                # Update covariance matrix
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * C)
                
                for i in range(mu):
                    diff = (x_sorted[i] - m_new) / sigma
                    C += c_mu * weights[i] * np.outer(diff, diff)
                
                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    # Clip small eigenvalues to avoid numerical issues
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C = (eigvecs * eigvals) @ eigvecs.T
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    # Reset if covariance matrix breaks
                    B = np.eye(d)
                    D = np.ones(d)
                    C = np.eye(d)
                    break # Restart this CMA-ES instance
                
                # Update pc
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ y)
                
                m = m_new
                
                # Check for stagnation
                if gen - last_improvement_gen > stagnation_limit:
                    break
            
            restart_count += 1
            # Decay initial sigma for next restart to focus on finer scales if needed, 
            # but IPOP usually increases or resets. Let's keep it constant or slightly increase 
            # to ensure we can escape local minima.
            # Actually, for Lunacek, if we are stuck in a local min, we need to jump out.
            # The reset logic above (m = best_x + noise) helps.
            
            # If we have leftover budget, do a (1+1)-ES step
            if budget > 0 and budget < lambda_:
                for _ in range(budget):
                    if time.time() - start_time > timeout:
                        break
                    y = best_x + np.random.randn(d) * 0.5
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    budget -= 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
