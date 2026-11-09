"""Hybrid Global-Local Optimizer: Sobol Global Search + Adaptive (1+1)-ES for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        
        # Phase 1: Global Search using Sobol Sequence
        # Allocate ~40% of budget to global exploration
        global_budget = int(total_budget * 0.4)
        
        # Generate Sobol points in [0,1]^d
        # We need enough points to cover the space reasonably
        # For d=5, 200 points is a good start. We will adaptively add more if time allows.
        sobol_gen = qmc.Sobol(d=d, scramble=True)
        # Get a large batch of points, we will use them as needed
        # 2^9 = 512 points is plenty for d=5
        sobol_points = sobol_gen.random(512) 
        
        # Map [0,1]^d to [-5,5]^d
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Evaluate Sobol points
        for i in range(min(global_budget, len(sobol_points))):
            if calls >= total_budget or time.time() > deadline:
                break
            x = sobol_points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # If we didn't use all global budget, we could continue, but let's switch to local
        # Actually, let's be more dynamic: if we have time and budget, use remaining for local
        
        # Phase 2: Local Refinement using Adaptive (1+1)-ES with Covariance Adaptation
        # We use a simplified CMA-ES-like update for a single point (1+1)
        # This is much faster than population-based CMA-ES
        
        # Initialize local state
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size
        C = np.eye(d)  # Covariance matrix
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Adaptation rates for (1+1)-ES (mu=1)
        # Standard CMA-ES parameters adapted for mu=1
        mu_eff = 1.0
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 2)) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        
        # Precompute B and D for Cholesky decomposition
        B = np.linalg.cholesky(C)
        D = np.prod(np.sqrt(np.diag(C)))
        
        # Local search loop
        # We will continue until budget or time runs out
        while calls < total_budget and time.time() < deadline:
            # Generate a candidate
            z = np.random.standard_normal(d)
            y = B @ z
            x_candidate = x_mean + sigma * y
            x_candidate = np.clip(x_candidate, -5, 5)
            
            fx_candidate = func(x_candidate)
            calls += 1
            
            # Update best
            if fx_candidate < best_f:
                best_f = fx_candidate
                best_x = x_candidate.copy()
            
            # Update local state (1+1 ES logic)
            # If candidate is better, move mean towards it
            if fx_candidate < best_f or (fx_candidate == best_f and np.random.rand() < 0.5):
                # Success
                # Update mean
                x_mean = x_candidate.copy()
                
                # Update covariance (rank-one update)
                # For (1+1), we use the successful step y
                # h_sigma calculation (simplified for mu=1)
                norm_y = np.linalg.norm(y)
                threshold = 1.4 + 2.0 / (d + 1)
                if norm_y < threshold * np.sqrt(d):
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0
                    
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                
                delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c)
                
                # C update
                C = (1 + c_1 - c_1) * C + c_1 * (np.outer(p_c, p_c) - C - delta_h)
                C = (C + C.T) / 2.0
                
                # Regularize
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-12:
                        C += (1e-12 - np.min(eigvals)) * np.eye(d)
                except:
                    C += 1e-12 * np.eye(d)
                
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(d)
                    B = np.eye(d)
                
                D = np.prod(np.sqrt(np.diag(C)))
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1.0))
                sigma = np.clip(sigma, 1e-12, 5.0)
            else:
                # Failure: do not move mean, but still update path memories slightly?
                # In standard (1+1), we only update on success.
                # However, to adapt to the landscape, we might want to shrink sigma if we fail repeatedly.
                # The standard CMA-ES sigma update handles this via the norm of p_sigma.
                # Since we don't update p_sigma on failure, norm(p_sigma) will decay, 
                # causing sigma to shrink if we have many failures.
                pass
            
            # Early termination if sigma becomes too small
            if sigma < 1e-8:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
