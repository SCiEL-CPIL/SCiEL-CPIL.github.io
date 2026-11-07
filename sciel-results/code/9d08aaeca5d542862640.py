"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        # Initial center: 0 is a safe guess for BBOB functions, 
        # but randomization helps if the shift is large.
        # The task says minimizer is in [-4, 4]^5.
        # We start at 0.
        m = np.zeros(self.dim)
        
        # Initial sigma: 
        # BBOB functions usually have a scale of 1. 
        # A larger sigma helps global exploration initially.
        sigma = 1.0
        
        # Population size
        # Standard lambda is 4 + floor(3 * log(dim))
        # For dim=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8.
        # Let's use a slightly larger population for better initial coverage.
        lam = 12
        
        # Weights
        # Standard weights: mu = lam // 2
        # w_i = log(mu + 0.5) - log(i)
        # c_mu = (mu_eff - 2) / ((n + 2)^2 + mu_eff)
        # c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        # c_1 = 2 / ((n + 1.3)^2 + mu_eff)
        # c_sigma = (mu_eff/2 + 1) / (n + 5)
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff / 2 + 1) / (self.dim + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, (mu_eff - 2) / ((self.dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        # State variables
        pc = np.zeros(self.dim)
        hs = np.zeros(self.dim)
        C = np.eye(self.dim)
        
        best_x = m.copy()
        best_f = np.inf
        calls_made = 0
        
        # Precompute square root of C for sampling
        # We will update C each generation
        
        while calls_made < self.budget:
            # Generate population
            pops = []
            fs = []
            
            # Check if we can afford a full generation
            # If not, we might need to adjust lambda or stop
            # For simplicity, we run full generations as long as possible.
            # If remaining budget < lam, we can still evaluate the rest.
            
            for i in range(lam):
                # Sample from N(m, sigma^2 C)
                # Use Cholesky or Eigen decomposition
                # For dim=5, eigen is fast and stable
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    # Ensure non-negative eigenvalues
                    eigvals = np.clip(eigvals, 1e-12, None)
                    sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals))
                except np.linalg.LinAlgError:
                    # Fallback to identity if decomposition fails
                    sqrt_C = np.eye(self.dim)
                
                z = rng.standard_normal(self.dim)
                y = m + sigma * (sqrt_C @ z)
                
                # Clip to bounds? 
                # The task says func(x) clips to [-5, 5].
                # It is better to let func clip, or clip ourselves to avoid out-of-bounds issues if func doesn't handle it well?
                # Task: "func(x) clips a finite length-5 vector to the box". So we can pass unclipped, but clipping ourselves might be slightly more efficient or consistent.
                # Let's pass unclipped, let func handle it.
                
                fy = func(y)
                calls_made += 1
                pops.append(y)
                fs.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if calls_made >= self.budget:
                break
            
            # Sort population by fitness
            order = np.argsort(fs)
            pops = [pops[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, np.newaxis] * np.array(pops[:mu]), axis=0)
            
            # Update path length
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (m - m_old)
            
            # Update covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * pc * pc.T + c_mu * sum(w_i * y_i * y_i.T)
            
            # Calculate y_i vectors (steps)
            y_vectors = np.array(pops[:mu]) - m_old
            y_vectors = y_vectors / sigma # Normalized steps
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum(
                weights[:, np.newaxis, np.newaxis] * y_vectors[:, np.newaxis, :] * y_vectors[:, :, np.newaxis], 
                axis=0
            )
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update sigma
            # Hessian norm check
            # ||C|| / sqrt(n) vs chi_n
            # For dim=5, we can just use the norm of pc
            # hs = sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (m - m_old)
            # This is essentially the step in the pc direction
            # Standard CMA-ES sigma update:
            # sigma = sigma * exp(c_sigma / chi_n * (||pc|| / chi_n - 1))
            
            pc_norm = np.linalg.norm(pc)
            sigma *= np.exp(c_sigma / chi_n * (pc_norm / chi_n - 1))
            
            # Clamp sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Optional: Restart mechanism?
            # With budget 500 and lam=12, we get ~40 generations.
            # CMA-ES is usually good enough for dim=5 in 40 gens if started well.
            # If stuck, we might need a restart, but CMA-ES is robust.
            # Let's add a simple stagnation check.
            if best_f < -1e6 or np.isnan(best_f):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
