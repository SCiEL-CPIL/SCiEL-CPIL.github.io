"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        # Parameters for CMA-ES
        # Population size lambda
        lam = 4 + int(3 * np.log(self.dim))
        mu = lam // 2
        
        # Initial sigma
        sigma0 = 2.0
        x0 = np.zeros(self.dim)
        
        # CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = (2 / ((self.dim + 1.3)**2 + mu_eff)) * (1 - c_c)
        c_mu = (2 / ((self.dim + 2)**2 + mu_eff)) * (min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((self.dim + 2)**2 + mu_eff)))
        
        # Budget allocation strategy
        # We have 500 calls total.
        # A typical CMA-ES run needs ~200-300 calls to converge reasonably in 5D.
        # Let's do multiple restarts.
        # Start with one long run, then shorter restarts if time permits.
        
        def cma_es_run(x_init, sigma_init, max_iter):
            nonlocal best_x, best_f
            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            chi_n = np.sqrt(self.dim) * (1 + 1. / (4 * self.dim) - 1. / (21 * self.dim**2))
            
            # Pre-compute inverse sqrt of C for efficiency (approximate or full)
            # For 5D, full decomposition is fast enough.
            
            for it in range(max_iter):
                # Check time
                if time.time() > deadline:
                    return
                
                # Sample
                B, d = np.linalg.eigh(C)
                # Ensure positive definite
                B = np.clip(B, 1e-12, None)
                # sqrt(C) = B * sqrt(d) * B.T
                # We need random vector z ~ N(0, I)
                # y = x_mean + sigma * B * sqrt(d) * z
                z = rng.standard_normal((lam, self.dim))
                # Project
                y = x_mean[None, :] + sigma * (z @ (B * np.sqrt(d)).T)
                
                # Clip to box
                y = np.clip(y, -5, 5)
                
                # Evaluate
                fvals = np.array([func(yp) for yp in y])
                
                # Update best
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = y[idx_min].copy()
                    
                # Sort
                order = np.argsort(fvals)
                y_sorted = y[order]
                f_sorted = fvals[order]
                
                # Update mean
                x_mean_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
                
                # Update covariance adaptation state
                y_inv_C = np.linalg.solve(C, (y_sorted[:mu] - x_mean) / sigma)
                pc = (1 - c_sigma) * pc + c_sigma * np.sum(weights[:, None] * y_inv_C, axis=0)
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1)))) < (1.4 + 2 / (self.dim + 1)) * chi_n
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * hsig * np.linalg.solve(C, (x_mean_new - x_mean) / sigma)
                
                # Update C
                C = (1 - c_1 - c_mu + (1 - c_1 - c_mu) * (1 - hsig)**2 * c_sigma * (2 - c_sigma)) * C \
                    + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_sigma * (2 - c_sigma) * C) \
                    + c_mu * np.sum(weights[:, None, None] * np.outer(y_inv_C, y_inv_C), axis=0)
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                # Regularize if needed
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    if np.min(eigvals) < 1e-12:
                        C = eigvecs @ np.diag(np.clip(eigvals, 1e-12, None)) @ eigvecs.T
                except:
                    C = np.eye(self.dim)
                    
                x_mean = x_mean_new
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Clamp sigma
                if sigma < 1e-12:
                    break
                if sigma > 1e6:
                    sigma = 1e6

        # Restart strategy
        # We have 500 budget.
        # Let's try 5 restarts of 100 iterations each? Or adaptive?
        # CMA-ES is expensive per iteration (matrix ops).
        # Let's estimate cost. 5D, lam=13. 13 func calls per iter.
        # 100 iters = 1300 calls. Too many.
        # We only have 500 calls total.
        # So max ~38 iterations per run.
        # Let's do 3 runs of 33 iters (33*13=429 calls) leaving buffer.
        # Or 1 run of 38 iters?
        # Multi-restart is better for multimodal.
        
        # Let's use a fixed number of restarts based on budget.
        # lam = 13.
        # Iters per restart = 500 // lam // num_restarts
        # Let's try 4 restarts. 500 // 13 // 4 = 9 iters. Too few to converge.
        # Maybe smaller population?
        # For 5D, lam=13 is standard.
        # Let's reduce lam to 9? mu=4.
        # 500 // 9 = 55 iters total.
        # 2 restarts of 27 iters?
        
        # Let's stick to standard CMA-ES params but limit iterations.
        # lam = 4 + int(3*log(5)) = 4 + 5 = 9.
        # mu = 4.
        
        lam = 9
        mu = 4
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = (2 / ((self.dim + 1.3)**2 + mu_eff)) * (1 - c_c)
        c_mu = (2 / ((self.dim + 2)**2 + mu_eff)) * (min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((self.dim + 2)**2 + mu_eff)))
        
        # Number of restarts
        # Each iteration costs 'lam' calls.
        # Total calls = num_restarts * iters_per_restart * lam
        # We want to use ~480 calls to be safe.
        # 480 // 9 = 53 iterations total.
        # Let's do 2 restarts of 26 iterations.
        # Or 1 restart of 53?
        # Multi-restart is safer for f22 (21 peaks).
        # Let's do 3 restarts of 17 iterations.
        
        num_restarts = 3
        iters_per_restart = (self.budget - 10) // lam // num_restarts
        # Ensure at least 10 iters per restart
        if iters_per_restart < 10:
            iters_per_restart = 10
            num_restarts = (self.budget - 10) // lam // iters_per_restart
            if num_restarts < 1:
                num_restarts = 1
                iters_per_restart = (self.budget - 10) // lam

        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization for restart
            x_init = rng.uniform(-4, 4, self.dim)
            sigma_init = 1.5
            
            cma_es_run(x_init, sigma_init, iters_per_restart)
            
            # If we found a very good solution, we might stop early?
            # No, keep restarting to explore other peaks.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
