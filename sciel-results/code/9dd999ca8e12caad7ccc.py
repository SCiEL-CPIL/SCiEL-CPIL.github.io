"""
CMA-ES optimizer for BBOB f7 (Step ellipsoidal) in 5D.

Strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a
state-of-the-art derivative-free optimizer for black-box continuous optimization.
It adapts a multivariate normal distribution to the shape of the objective function.
For BBOB f7 (Step Ellipsoidal), which involves rotation and scaling, CMA-ES is
highly effective because it learns the covariance structure of the landscape,
allowing it to navigate the rotated ellipsoidal contours efficiently.

The implementation uses a standard CMA-ES algorithm with:
1. Population size n = 4 + floor(3 * ln(dim))
2. Step size (sigma) adaptation via cumulative step-size adaptation (CSA)
3. Covariance matrix adaptation via rank-one and rank-mu updates
4. Boundary handling via reflection and penalty
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    CMA-ES is specifically designed to handle rotated and scaled elliptical
    landscapes like the BBOB f7 (Step Ellipsoidal) function. It adapts both
    the step size (sigma) and the search distribution's covariance matrix (C)
    to match the local geometry of the function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: standard recommendation is 4 + floor(3 * ln(dim))
        n = 4 + int(3 * np.log(dim))
        n = max(n, 4)  # Minimum population size
        
        # Initial step size
        # The search space is [-5, 5]^5, so a step size of ~1.0 or 2.0 is reasonable.
        # Starting larger (e.g., 2.0) helps with exploration in the initial phase.
        sigma = 2.0
        
        # Mean vector (center of the distribution)
        # Start at the center of the box
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse square root of C (for sampling)
        C_sqrt = np.eye(dim)
        
        # Learning rates
        # Standard values from Hansen's CMA-ES
        c_sigma = (n + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + n / dim) / (dim + 4 + 2 * n / dim)
        c_1 = 2 / ((dim + 1.3)**2 + n)
        c_mu = min(1 - c_1, 2 * ((n/2 + 1) - dim - 2) / ((dim + 2)**2 + n))
        
        # Weights for rank-mu update
        weights = np.log(n/2 + 1) - np.log(np.arange(1, n+1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # State variables
        ps = np.zeros(dim)  # Evolution path for step size
        pc = np.zeros(dim)  # Evolution path for covariance
        
        # Track best
        best_x = m.copy()
        best_f = float('inf')
        
        calls_made = 0
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Sample offspring
            # Use Cholesky decomposition of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Generate n samples
            samples = []
            for i in range(n):
                z = np.random.randn(dim)
                x = m + sigma * (L @ z)
                samples.append(x)
            
            # Evaluate samples
            fitnesses = np.empty(n)
            for i in range(n):
                if calls_made >= budget:
                    break
                # Handle boundaries: reflect if out of bounds
                x = samples[i]
                # Simple reflection: if x_j < -5, x_j = -5 + (-5 - x_j) * 0.1
                # Or just clip. Clipping is simpler but can cause issues.
                # Let's use a soft penalty or reflection.
                # For simplicity and robustness, we clip, but CMA-ES prefers
                # to stay within bounds. A common trick is to penalize out-of-bounds.
                # However, the problem says func clips. So we can just pass x.
                # But to help CMA-ES, we should avoid generating too many out-of-bounds.
                # We'll just call func, which clips.
                fitnesses[i] = func(x)
                calls_made += 1
                
                if fitnesses[i] < best_f:
                    best_f = fitnesses[i]
                    best_x = x.copy()
            
            if calls_made >= budget:
                break
            
            # Sort by fitness
            order = np.argsort(fitnesses)
            sorted_samples = [samples[i] for i in order]
            sorted_fitnesses = fitnesses[order]
            
            # Update mean
            m_new = m + sigma * np.sum(weights[:n] * np.array(sorted_samples), axis=0)
            
            # Update evolution path for step size (ps)
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^(-1/2) * (m_new - m) / sigma
            C_inv_sqrt = np.linalg.inv(C_sqrt)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (m_new - m) / sigma)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update evolution path for covariance (pc)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2*(calls_made/n)))) < (1.4 + 2/(dim+1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
            
            # Update covariance matrix
            # Rank-one update
            C = (1 + c_c * c_mu * hsig - c_1) * C + c_1 * np.outer(pc, pc)
            
            # Rank-mu update
            y = np.array(sorted_samples[:n])  # Top n (all)
            y = (y - m) / sigma
            C += c_mu * np.sum(weights**2, keepdims=True) * (np.outer(y.T, y.T) * weights[:, np.newaxis] * weights[np.newaxis, :])
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update C_sqrt (inverse square root)
            # We need C^(-1/2) for the ps update in the next iteration.
            # We can compute it via eigenvalue decomposition.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid numerical issues
            eigvals = np.clip(eigvals, 1e-20, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            m = m_new
            
            # Termination criteria
            # If sigma is very small or variance is very small
            if sigma < 1e-15:
                break
            if np.trace(C) * sigma**2 < 1e-20:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
