"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.call_count = 0
        self.start_time = None

    def _eval(self, func, x):
        """Wrapper to track budget and time."""
        if self.call_count >= self.budget:
            return self.best_f
        self.call_count += 1
        # Deadline check
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > 280: # Safety margin
                return self.best_f
        
        fx = func(x)
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def _lhs_sample(self, n, dim, rng):
        """Generate n samples using Latin Hypercube Sampling in [-5, 5]^dim."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        # Map to [-5, 5]
        samples = samples * 10.0 - 5.0
        return samples

    def _cmaes(self, func, x0, sigma, budget_left):
        """
        Compact CMA-ES implementation.
        Returns best_x, best_f, calls_used.
        """
        n = self.dim
        N = 4 + int(3 * np.log(n))  # Population size, e.g., 10 for n=5
        mu = N // 2                  # Number of parents
        # Weights: log-linear
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State
        mean = x0.copy()
        C = np.eye(n)
        C_1 = np.sqrt(C) # Cholesky or Sqrt
        # For simplicity and stability in compact implementation, we update C via eigendecomposition
        # which is O(n^3) but n=5 is tiny.
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        calls = 0
        best_x = mean.copy()
        best_f = self._eval(func, mean)
        calls += 1
        
        while calls < budget_left:
            # Generate population
            W = np.zeros((N, n))
            F = np.zeros(N)
            
            for i in range(N):
                # Sample
                z = rng.standard_normal(n)
                # Apply C_1 (sqrt of covariance)
                # We maintain C_1 such that C_1 @ C_1.T = C
                # Update C_1 after C update
                y = C_1 @ z
                x_new = mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = self._eval(func, x_new)
                calls += 1
                W[i] = x_new
                F[i] = f_new
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    
                # Time check inside loop
                if self.start_time is not None and (time.time() - self.start_time > 280):
                    return best_x, best_f, calls

            # Sort
            idx = np.argsort(F)
            W = W[idx]
            F = F[idx]
            
            # Update mean
            mean_old = mean.copy()
            mean += sigma * np.sum(w[:, np.newaxis] * W[:mu], axis=0)
            
            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / C_1 @ (mean - mean_old) / sigma
            # Check if ps is out of bounds
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1) / N)) > np.sqrt(n) + 1.4 + 2 / n:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / C_1 @ (mean - mean_old) / sigma
            else:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma
            
            # Update covariance matrix
            delta_h = (1 + 2 * max(0, (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(calls + 1) / N) - 1) / (n + 1))) * (1 - c_1)
            
            rank_1 = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (W[i] - mean_old) / sigma
                rank_mu += w[i] * np.outer(y_i, y_i)
                
            C = (1 + c_1 * (delta_h - 1) - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Eigendecomposition to update C_1 and fix numerical issues
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_1 = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Adapt sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            # Prevent sigma from becoming too large or small
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Convergence check
            if sigma < 1e-10 or np.max(eigvals) / np.min(eigvals) > 1e12:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(0)
        
        # Phase 1: LHS Sampling
        n_lhs = 50
        if self.budget < n_lhs + 10:
            n_lhs = max(5, self.budget // 2)
            
        lhs_samples = self._lhs_sample(n_lhs, self.dim, rng)
        
        for x in lhs_samples:
            if self.call_count >= self.budget:
                break
            self._eval(func, x)
            
        # If we ran out of budget during LHS, return best
        if self.call_count >= self.budget:
            return self.best_x, self.best_f

        # Phase 2: CMA-ES
        # Start from best point found in LHS
        x0 = self.best_x.copy()
        # Initial sigma: heuristic based on box size or fixed
        sigma0 = 1.0 
        
        remaining_budget = self.budget - self.call_count
        best_x, best_f, calls_used = self._cmaes(func, x0, sigma0, remaining_budget)
        
        # Update global best
        if best_f < self.best_f:
            self.best_f = best_f
            self.best_x = best_x
            
        # Phase 3: Polish with Nelder-Mead if we have some budget left and haven't converged
        # CMA-ES usually converges well, but NM can squeeze out the last bit.
        # However, NM uses multiple function calls per iteration.
        # Let's check if we have at least 20 calls left.
        if self.call_count < self.budget - 20 and self.best_f > -1e8:
            # Use scipy Nelder-Mead
            # Note: minimize uses its own function calls, we need to wrap it to count them
            # But minimize might exceed budget if not careful.
            # Simple approach: Just run a few manual Nelder-Mead steps or trust CMA-ES.
            # Given the budget is small, and CMA-ES is strong, let's stick to CMA-ES for simplicity and robustness.
            pass

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
