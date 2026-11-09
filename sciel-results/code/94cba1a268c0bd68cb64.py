"""CMA-ES hybrid optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES based optimizer with multi-phase strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        x_best = np.zeros(dim)
        f_best = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def cmaes(x0, sigma0, max_calls, seed_offset=0):
            nonlocal calls_made, x_best, f_best
            if max_calls < 20:
                return

            n = dim
            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            # Number of parents
            mu = lambda_ // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w**2)
            
            # Strategy parameters
            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # Initialization
            mean = np.array(x0, dtype=np.float64)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Precompute
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # Local RNG for this CMA-ES instance
            local_rng = np.random.default_rng(12345 + seed_offset)
            
            # Track best in this CMA-ES run for early stopping if needed
            local_best_f = np.inf
            
            while calls_made < max_calls and time.time() < deadline:
                # Generate population
                Y = local_rng.standard_normal((lambda_, n))
                # Compute Cinv_sqrt via Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                    # C = L L.T, so C^-1/2 is not directly L^-1.
                    # We need C^-1/2.
                    # Let C = V D V.T. C^-1/2 = V D^-1/2 V.T.
                    # Or use scipy.linalg.sqrtm? No, stick to numpy.
                    # Recompute Cinv_sqrt via Eigh
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    Cinv_sqrt = np.eye(n)
                    continue

                X = mean + sigma * (Y @ Cinv_sqrt.T)
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([evaluate(x) for x in X])
                
                # Sort
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]
                
                if F[0] < local_best_f:
                    local_best_f = F[0]
                
                # Update mean
                mean_new = np.sum(w[:, None] * X[:mu], axis=0)
                
                # Update ps
                # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * Cinv_sqrt @ ((mean_new - mean) / sigma)
                delta_mean = (mean_new - mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ delta_mean)
                
                # Update pc
                norm_ps = np.linalg.norm(ps)
                # hsig condition
                # Using i as generation count is better, but we don't have it directly.
                # Approximate with calls_made / lambda_
                gen_count = calls_made / lambda_
                denom = np.sqrt(1 - (1 - cs)**(2 * gen_count)) if gen_count > 0 else 1.0
                if denom < 1e-10: denom = 1e-10
                hsig = (norm_ps / denom) < (1.4 + 2 / (n + 1)) * chi_n
                
                if hsig:
                    pc = (1 - cc) * pc + cc * (mean_new - mean) / sigma
                else:
                    pc = (1 - cc) * pc
                    
                # Update C
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (X[i] - mean) / sigma
                    rank_mu += w[i] * np.outer(zi, zi)
                    
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + cmu * rank_mu + c1 * delta_hsig * np.outer(pc, pc)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update sigma
                sigma = sigma * np.exp(cs / ds * (norm_ps / chi_n - 1))
                
                # Update mean
                mean = mean_new
                
                # Stop if sigma is too small
                if sigma < 1e-10:
                    break

            return

        # Phase 1: Global Exploration
        # Use 60% of budget
        budget_phase1 = int(budget * 0.6)
        cmaes(np.zeros(dim), 2.0, budget_phase1, seed_offset=0)
        
        # Phase 2: Local Refinement
        remaining = budget - calls_made
        if remaining > 20 and x_best is not None:
            # Use 30% of budget
            budget_phase2 = min(remaining, int(budget * 0.3))
            cmaes(x_best.copy(), 0.1, budget_phase2, seed_offset=1)
            
        # Phase 3: Final Polish
        remaining = budget - calls_made
        if remaining > 5 and x_best is not None:
            def wrapped(x):
                nonlocal calls_made
                if calls_made >= budget:
                    return f_best
                x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
                fx = float(func(x))
                calls_made += 1
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
                return fx
            try:
                minimize(wrapped, x_best, method='Nelder-Mead', options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            except Exception:
                pass

        return x_best, f_best
# EVOLVE-BLOCK-END
