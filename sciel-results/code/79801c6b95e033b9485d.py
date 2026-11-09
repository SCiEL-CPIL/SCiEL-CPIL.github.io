"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with integrated restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Global best tracking
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        
        # CMA-ES parameters
        # Population size lambda
        lam = 4
        # Initial sigma
        sigma = 2.0
        # Initial mean
        m = x_best.copy()
        # Initial covariance matrix
        C = np.eye(self.dim)
        # Inverse square root of C (for sampling)
        B = np.eye(self.dim)
        D = np.ones(self.dim)
        # Parameters for adaptation
        cm = 1.0 / (self.dim + 3)
        cs = 1.0 / (self.dim + 5)
        c1 = 2 / ((self.dim + 1.3)**2 + lam)
        cmu = min(0.5 - c1, 2 * (2 - (self.dim + 2) / (lam + 4)) / ((lam + 2)**2 + self.dim + 2))
        # Path lengths
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        # Success rule for sigma
        # We use a simple success rule: if best improves, increase sigma slightly, else decrease.
        # Standard CMA-ES uses path length for sigma adaptation, but simple success rule is robust for small budgets.
        # Let's use the standard path-based sigma adaptation if possible, or a simple heuristic.
        # Given the tight budget (500 calls) and dim 5, a simplified CMA-ES or (mu+lambda) with covariance update is key.
        
        # Actually, implementing full CMA-ES with path adaptation is complex. 
        # A robust alternative for this specific "black-box" task with limited budget is often a 
        # well-tuned (1+1)-ES with restarts or a simple CMA-ES variant.
        # The prompt asks for CMA-ES. Let's implement a standard CMA-ES loop.
        
        # Precompute constants for sigma adaptation
        # d_sigma
        d_sigma = 1.0 + 2 * max(0, (self.dim / lam - 1) / (self.dim + 2))
        
        # Norm of expected length of standard normal vector
        chi_lam = np.sqrt(lam) * (1 - 1/(4*lam) + 1/(21*lam**2))
        
        calls_made = 1
        x_history = [x_best]
        
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # 1. Sample lambda points
            # y_i = B @ (D * z_i)
            # z_i ~ N(0, I)
            Z = rng.standard_normal((lam, self.dim))
            Y = Z @ np.diag(D) @ B.T
            # x_i = m + sigma * y_i
            X = np.array([m + sigma * y for y in Y])
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # 2. Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            lam_eff = lam // 2  # mu = lam/2 typically
            
            # 3. Update mean
            weights = np.ones(lam_eff) / lam_eff
            m_new = np.sum(weights[:, np.newaxis] * X[:lam_eff], axis=0)
            
            # 4. Update covariance matrix
            # y_i = (x_i - m_old) / sigma
            Y_old = (X[:lam_eff] - m) / sigma
            C_new = C
            for i in range(lam_eff):
                C_new += cmu * weights[i] * np.outer(Y_old[i], Y_old[i])
            C = C_new
            
            # 5. Update step size sigma
            # Path evolution strategy
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*(1+cmu*2))*delta_s*B^{-1}C^{-1/2}m_new-m
            # m_new - m is in y-space? No, x-space.
            # delta_s = 1 if best < prev_best else 0
            best_f_curr = F[0]
            prev_best_f = f_best
            delta_s = 1.0 if best_f_curr < prev_best_f else 0.0
            
            # We need B^{-1} C^{-1/2} which is essentially the inverse of the transformation used in sampling.
            # Since X = m + sigma * B * D * Z, the inverse transformation for a vector v in x-space to z-space is B^{-1} D^{-1} v.
            # However, the standard CMA-ES update for ps uses the inverse of the square root of C.
            # Let's simplify: use a simple success rule for sigma to avoid complex matrix inversions which might be slow or error-prone in a tight loop.
            # If best improves, sigma *= 1.1, else sigma *= 0.9.
            # This is less theoretically sound but often effective for small dims and budgets.
            if best_f_curr < f_best:
                f_best = best_f_curr
                x_best = X[0].copy()
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            m = m_new
            
            # Update B and D (eigendecomposition) - this is expensive.
            # For dim=5, it's cheap. But doing it every generation might be overkill or slow.
            # Let's do it every 2 generations or if C changes significantly.
            # Actually, for dim 5, eigendecomposition is very fast.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Handle negative eigenvalues due to numerical errors
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Ensure m stays within bounds? CMA-ES usually doesn't clip m, but samples are clipped.
            # The problem states func clips x. So we don't need to clip m or X explicitly, func handles it.
            # However, keeping m reasonable helps.
            
            # Check for convergence/stagnation
            # If sigma becomes very small or C becomes singular, restart.
            if sigma < 1e-4 or np.any(eigvals < 1e-12):
                # Restart
                m = x_best + rng.uniform(-0.5, 0.5, self.dim)
                sigma = 1.0
                C = np.eye(self.dim)
                B = np.eye(self.dim)
                D = np.ones(self.dim)
                ps = np.zeros(self.dim)
                pc = np.zeros(self.dim)
                
        return x_best, f_best
# EVOLVE-BLOCK-END
