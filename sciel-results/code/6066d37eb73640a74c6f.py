"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Population-based CMA-ES for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    - Full CMA-ES implementation to handle rotation and non-separability.
    - Large initial sigma to cover the search space quickly.
    - Strict budget and time management.
    - Fallback to (1+1)-ES if CMA-ES stalls or budget is tight.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, x0, sigma, func, budget, deadline, rng):
        """
        Run CMA-ES starting from x0 with initial sigma.
        Returns (best_x, best_f, calls_made).
        """
        n = self.dim
        # CMA-ES parameters
        # Default lambda
        lam = int(4 + 3 * np.log(n))
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2 / mueff) - 2) / ((n + 2)**2 + mueff))
        
        # Step size control
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initial state
        x_mean = x0.copy()
        sigma = sigma
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = (c_c + 1) / (n + c_c + 1)
        hs = (1 + c_sigma) / (1 - c_sigma)
        
        best_x = x0.copy()
        best_f = func(x0)
        calls_made = 1
        
        # Initial population
        # We need to evaluate the initial population to get the first set of samples
        # But to save budget, we can just start the loop and evaluate in the loop.
        # However, standard CMA-ES samples lambda points per generation.
        
        while calls_made < budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            if calls_made + lam > budget:
                # Not enough for a full generation, maybe do a partial one or break
                # For simplicity, break if we can't do a full generation to maintain algorithm integrity
                # Or we can reduce lam? Let's just break.
                break
            
            # Sample population
            z = rng.standard_normal((lam, n))
            # Transform to x
            # C^{1/2} is expensive, use Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
            
            pop = x_mean + sigma * (z @ L.T)
            
            # Evaluate population
            fvals = np.array([func(x) for x in pop])
            calls_made += lam
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx].copy()
            
            # Sort by fitness
            order = np.argsort(fvals)
            y = pop[order]
            fvals_sorted = fvals[order]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * y[:mu], axis=0)
            
            # Update step size control
            # Norm of (x_mean - x_mean_old) / sigma
            # Note: x_mean - x_mean_old = sigma * C^{1/2} * (sum w_i z_{i,sorted})
            # Let z_mean = sum w_i z_{i,sorted}
            z_mean = np.sum(weights[:, None] * z[order][:mu], axis=0)
            
            # Expected norm of standard normal vector
            # chi_n = sqrt(n * (1 - 1/(4n) + 1/(21n^2)))
            chi_n = np.sqrt(n * (1 - 1.0/(4.0*n) + 1.0/(21.0*n*n)))
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / np.linalg.norm(L) * z_mean
            # Wait, the update rule for ps is:
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * C^{-1/2} (xmean - xmean_old)/sigma
            # Since xmean - xmean_old = sigma * C^{1/2} * z_mean, then C^{-1/2}(...) = z_mean
            # So ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * z_mean
            
            # Correcting the previous line:
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_mean
            
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lam))) < (1.4 + 2 / (n + 1)) * chi_n:
                pc = (1 - c_c) * pc + c_c * (x_mean - x_mean_old) / sigma
            
            # Update covariance
            delta_hs = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lam))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            
            rank_mu = np.sum(weights[:, None] * (y[:mu] - x_mean_old)[:, None] * (y[:mu] - x_mean_old)[None, :], axis=0) / sigma**2
            
            C = (1 - c_1 - c_mu + c_1 * delta_hs) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clip sigma
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Check for stagnation or convergence
            if np.linalg.norm(x_mean - x_mean_old) < 1e-10 * max(1, sigma):
                break
        
        return best_x, best_f, calls_made

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320  # Safety margin
        rng = np.random.default_rng(42)
        
        self._best_x = None
        self._best_f = float('inf')
        calls_made = 0
        
        # Phase 1: CMA-ES
        # Start near origin, large sigma
        x0 = np.zeros(self.dim)
        sigma0 = 2.0  # Large initial step to explore
        
        # Run CMA-ES with most of the budget
        # Leave some budget for potential restarts or refinement
        budget_cma = int(self.budget * 0.8)
        
        try:
            best_x, best_f, cma_calls = self._cmaes(x0, sigma0, func, budget_cma, deadline, rng)
            calls_made += cma_calls
            self._best_x = best_x
            self._best_f = best_f
        except Exception:
            pass
        
        # Phase 2: If we have budget left, do a quick (1+1)-ES refinement
        if calls_made < self.budget and time.time() < deadline and self._best_x is not None:
            x = self._best_x.copy()
            fx = self._best_f
            sigma = 0.5
            
            while calls_made < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < self._best_f:
                        self._best_f = fx
                        self._best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if sigma < 0.001:
                        break
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
