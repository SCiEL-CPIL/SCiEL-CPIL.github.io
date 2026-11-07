import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f14.
    CMA-ES is robust for continuous, non-linear optimization and adapts to the 
    local geometry of the function, which is crucial for 'Different powers' (f14).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        N = 4 + int(3 * np.log(dim))
        if N < 10: N = 10
        if N > 20: N = 20 # Cap for small dim to save budget
        
        # Weights
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        rng = np.random.default_rng(42)
        # Start near center, but allow some spread
        mean = rng.uniform(-2.0, 2.0, dim)
        C = np.eye(dim)
        sigma = 2.0
        
        best_x = mean.copy()
        best_fx = float('inf')
        total_evals = 0
        
        def evaluate(x):
            nonlocal total_evals, best_x, best_fx
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            total_evals += 1
            if f < best_fx:
                best_fx = f
                best_x = x.copy()
            return f

        # Initial evaluation
        fx = evaluate(mean)
        
        # CMA-ES Loop
        while total_evals < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample candidates
            # Cholesky decomposition for sampling from N(0, C)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular
                C = np.eye(dim)
                L = np.eye(dim)
                
            candidates = []
            for i in range(N):
                z = rng.standard_normal(dim)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5.0, 5.0)
                candidates.append(x)
            
            # 2. Evaluate candidates
            fvals = np.array([evaluate(x) for x in candidates])
            
            # 3. Sort by fitness
            order = np.argsort(fvals)
            candidates = [candidates[i] for i in order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = candidates[0].copy()
            
            # 4. Update mean
            new_mean = np.sum([weights[i] * candidates[i] for i in range(mu)], axis=0)
            
            # 5. Update step size
            # Path for step size control
            y_sigma = (new_mean - mean) / sigma
            # Expected norm of standard normal
            E_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            # We need to accumulate p_sigma
            # Standard CMA-ES uses a path variable p_sigma
            # To keep it simple and robust, we can use a simplified update or full path.
            # Let's use full path variables for better performance.
            
            # Initialize path variables if not defined (first iteration)
            if not hasattr(self, '_p_sigma'):
                self._p_sigma = np.zeros(dim)
                self._p_c = np.zeros(dim)
            
            self._p_sigma = (1 - c_sigma) * self._p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(self._p_sigma) / E_norm - 1))
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # 6. Update covariance
            # Path for covariance control
            self._p_c = (1 - c_c) * self._p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            h_sigma = 1 if np.linalg.norm(self._p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / E_norm < (1.4 + 2/(dim+1)) * np.sqrt(1 - (1 - c_c)**2) else 0
            
            # Rank-1 update
            C = (1 - c_c * h_sigma) * C + c_c * (self._p_c @ self._p_c.T - (1 - h_sigma) * C)
            
            # Rank-mu update
            # y_i = (x_i - mean) / sigma
            ys = np.array([(candidates[i] - mean) / sigma for i in range(mu)])
            rank_mu_update = np.sum([weights[i] * (ys[i] @ ys[i].T) for i in range(mu)], axis=0)
            C += c_mu * rank_mu_update
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
            
            mean = new_mean
            
            # Check for stagnation or convergence
            if sigma < 1e-6:
                break
                
        return best_x, best_fx
