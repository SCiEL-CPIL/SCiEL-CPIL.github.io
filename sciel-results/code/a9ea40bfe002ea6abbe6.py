"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        # CMA-ES parameters
        dim = self.dim
        mu = max(1, dim // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        x_best_global = None
        f_best_global = np.inf

        # Phase 1: Initial exploration with large step size
        # Phase 2: Standard CMA-ES
        # Phase 3: Final polish with small step size if time permits
        
        # We will run CMA-ES. 
        # Initial point: 0 is often good for shifted/rotated if shift is small, 
        # but random is safer. Let's use 0 for simplicity and symmetry, 
        # or maybe a random point. The task says shift is in [-4, 4]^5.
        # Let's start at 0.
        
        x = np.zeros(dim)
        f_x = func(x)
        if f_x < f_best_global:
            f_best_global = f_x
            x_best_global = x.copy()
            
        sigma = 2.0 # Initial step size
        
        # CMA-ES State
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = np.sqrt((dim + 2) * (c_sigma + 0.5))
        cc = (1 - c_c) * (1 + c_sigma)
        
        # Budget management
        # CMA-ES uses lambda = 2 * mu evaluations per generation.
        # Total budget is 500.
        # Generations approx 500 / (2 * mu).
        
        # We need to be careful with the number of calls.
        # Each generation: 1 (for x) is already done? No, x is the mean.
        # We evaluate lambda children.
        # So each generation costs lambda calls.
        
        lam = 2 * mu
        
        # Pre-allocate random numbers? No, just use rng.
        rng = np.random.default_rng(42)
        
        # We will run until budget or time limit.
        # Keep track of calls made.
        calls_made = 1 # for x=0
        
        # To maximize AOCC, we want fast initial progress.
        # CMA-ES is slow to start but robust.
        # Maybe a hybrid? 
        # Let's stick to pure CMA-ES but with a good initial sigma.
        # Bent Cigar: one dim is scaled by 10000.
        # Initial sigma 2.0 is reasonable for [-5, 5] box.
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            # Generate lambda samples
            # We need to compute B, D from C
            # Eigen decomposition is expensive for dim=5? No, it's small.
            # But we can use Cholesky if C is positive definite (it should be).
            # Let's use Cholesky for speed.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if numerical issues
                L = np.eye(dim)
            
            # Generate standard normal matrix
            Z = rng.standard_normal((lam, dim))
            # Transform
            Y = Z @ L.T # Y_i = Z_i @ L.T = (L @ Z_i.T).T
            # Wait, if C = L L.T, then x = x + sigma * L @ z
            # So Y_i = L @ Z_i.T
            Y = (L @ Z.T).T
            
            # Evaluate
            X = x + sigma * Y
            # Clip to box
            X = np.clip(X, -5, 5)
            
            F = np.array([func(xi) for xi in X])
            calls_made += lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < f_best_global:
                f_best_global = F[min_idx]
                x_best_global = X[min_idx].copy()
            
            # Sort
            sort_idx = np.argsort(F)
            Y_sorted = Y[sort_idx]
            F_sorted = F[sort_idx]
            
            # Selection
            Y_sel = Y_sorted[:mu]
            
            # Update mean
            x_new = x + sigma * (w @ Y_sel)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (w @ Y_sel).T)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * calls_made / lam)) < (1 + c_sigma) * np.sqrt(dim + 1) - 1)
            if hsig:
                pc = (1 - c_c) * pc + c_c * (x_new - x) / sigma
            
            # Update C
            delta_hsig = (1.0 if hsig else 0.0) - (1 - c_c)
            C = (1 + c_c * c_sigma * (2 - c_sigma) + delta_hsig * c_c) * C \
                - c_sigma * (C @ pc[:, None] @ pc[:, None].T / sigma**2 + pc[:, None] @ pc[:, None].T / sigma**2) \
                + c_sigma * (w @ (Y_sel @ Y_sel.T))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * calls_made / lam)) - np.sqrt(dim)))
            
            # Accept step
            x = x_new
            
            # Safety: if sigma becomes too small or too large, restart or adjust
            if sigma < 1e-10:
                break
            if sigma > 10:
                sigma = 10

        return x_best_global, f_best_global
# EVOLVE-BLOCK-END
