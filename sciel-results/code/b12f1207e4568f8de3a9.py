"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For n=5, lambda is typically 4 + floor(3*ln(n)) = 4 + 4 = 8, or just 10.
        # Let's use lambda = 10 for stability.
        lam = 10
        mu = lam // 2
        
        # Initialization
        x0 = np.zeros(n)
        sigma0 = 2.0  # Initial step size, box is [-5,5], so 2 is reasonable
        
        # Weights
        weights = np.log(np.arange(mu, 0, -1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4. + mueff / (n + 4.)) / ((n + 2.) + 4. + mueff / (n + 4.))
        cs = (mueff + 2.) / (n + mueff + 5.)
        c1 = 2. / ((n + 1.3)**2 + mueff)
        cmu = min(1. - c1, 2. * (mueff - 2. + 1. / mueff) / ((n + 2.)**2 + mueff))
        
        # Initial state
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute Cholesky factor of C (initially identity)
        # In a full CMA-ES, we update C and recompute Cholesky every generation.
        # For efficiency in Python, we might limit the number of generations or use a simplified update.
        # However, n=5 is small, so Cholesky is cheap.
        
        best_x = x0.copy()
        best_f = func(best_x)
        calls_used = 1
        
        # Determine number of generations
        # We have budget calls. Each generation uses lam calls.
        # We reserve some calls for final polishing if needed, but CMA-ES usually handles it.
        # Let's just run until budget is exhausted.
        
        while calls_used + lam <= budget:
            # Generate population
            pop_x = np.zeros((lam, n))
            pop_f = np.zeros(lam)
            
            # Sample from N(x, sigma^2 * C)
            # We need the square root of C.
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular, reset or regularize
                C = np.eye(n)
                C_sqrt = np.eye(n)
            
            for i in range(lam):
                z = np.random.randn(n)
                pop_x[i] = x + sigma * (C_sqrt @ z)
                # Clip to box [-5, 5]
                pop_x[i] = np.clip(pop_x[i], -5, 5)
                pop_f[i] = func(pop_x[i])
                calls_used += 1
                
                if pop_f[i] < best_f:
                    best_f = pop_f[i]
                    best_x = pop_x[i].copy()
            
            # Sort by fitness
            idx = np.argsort(pop_f)
            y = pop_x[idx][:mu]
            f_sorted = pop_f[idx][:mu]
            
            # Update mean
            x_new = np.sum(weights[:, None] * y, axis=0)
            
            # Update covariance matrix
            # pc = (1-cs)*pc + sqrt(cs*(2-cs)*mueff) * (x_new - x)/sigma
            ps = (1. - cs) * ps + np.sqrt(cs * (2. - cs) * mueff) * (x_new - x) / sigma
            
            # Heaviside function
            hsig = 1. if np.linalg.norm(ps) / np.sqrt(1. - (1. - cs)**2) < (1.4 + 2. / (n + 1.)) * np.sqrt(np.trace(C) / n) else 0.
            
            # Update C
            # C = (1-c1-cmu)*C + c1*(ps*ps.T + (hsig-1+cs*(1-cs))*C) + cmu*sum(wi*(yi-x)*(yi-x).T)
            delta_hsig = (1. - hsig) * cs * (1. - cs)
            C = (1. - c1 - cmu) * C + c1 * (np.outer(ps, ps) + delta_hsig * C)
            
            diff = y - x
            C += cmu * np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2.0
            
            # Update sigma
            # sigma = sigma * exp(cs/(2*n_sigma) * (||ps||/sqrt(n_sigma) - 1))
            n_sigma = (1. + 2. * max(0., np.sqrt((mueff + 1.) / (n + 1.) - 1.) - 1.))
            sigma = sigma * np.exp(cs / (2. * n_sigma) * (np.linalg.norm(ps) / np.sqrt(n_sigma) - 1.))
            
            # Update mean
            x = x_new
            
            # Optional: If sigma becomes too small, we might stop early or just let it run.
            # With 500 budget and lam=10, we get ~50 generations.
            
        # Final check: ensure we return the best found
        return best_x, best_f
# EVOLVE-BLOCK-END
