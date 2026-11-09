import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 285.0
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0
        
        # CMA-ES Parameters
        # lambda = population size
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 2
        lam = max(lam, 4) # Ensure at least 4
        # For dim=5, lam = 4 + 3*1.6 = 8.8 -> 8. Let's use 10 for more robustness.
        lam = 10 
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        
        mueff = np.sum(w) ** 2 / np.sum(w ** 2)
        
        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        
        # Initial state
        # Use a multi-start strategy? 
        # The previous program used multi-start NM. 
        # Let's try 2 restarts of CMA-ES.
        n_restarts = 2
        budget_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            if evals_used >= budget or time.time() > deadline:
                break
            
            # Initialize mean
            if r == 0:
                x_mean = np.zeros(dim)
            else:
                x_mean = rng.uniform(-4.5, 4.5, size=dim)
            
            sigma = 1.5 # Initial step size
            C = np.eye(dim) # Covariance matrix
            pc = np.zeros(dim) # Path for covariance adaptation
            ps = np.zeros(dim) # Path for sigma adaptation
            
            # Cholesky factor of C
            Cmat = C.copy()
            L = np.linalg.cholesky(Cmat + 1e-10 * np.eye(dim))
            
            # Run CMA-ES
            while evals_used + lam <= budget and time.time() < deadline:
                # Generate population
                zs = rng.standard_normal((lam, dim))
                ys = zs @ L.T
                xs = x_mean + sigma * ys
                xs = np.clip(xs, -5, 5)
                
                # Evaluate
                fs = np.zeros(lam)
                for i in range(lam):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    f_val = func(xs[i])
                    evals_used += 1
                    fs[i] = f_val
                    if f_val < best_f:
                        best_f = f_val
                        best_x = xs[i].copy()
                
                if evals_used < budget - lam + 1:
                    break
                
                # Sort
                idx = np.argsort(fs)
                xs = xs[idx]
                ys = ys[idx]
                fs = fs[idx]
                
                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
                
                # Update paths
                # pc = (1 - cs) * pc + sqrt(cs * (2 - cs) * mueff) / sigma * (C @ (x_mean - x_mean_old))
                # Note: C @ (x_mean - x_mean_old) is expensive if we don't use L.
                # Standard CMA-ES uses L^{-1} (x_mean - x_mean_old) for ps?
                # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * L^{-1} (x_mean - x_mean_old)
                # pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mueff) * (x_mean - x_mean_old)
                
                # Let's stick to standard formulas
                # ps update
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.solve(L.T, (x_mean - x_mean_old) / sigma)
                
                # hsig
                chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim) + 1 / (12 * dim ** 3))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_used // lam + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
                
                # pc update
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_mean - x_mean_old) / sigma
                
                # Update Covariance C
                delta = 1 if hsig else 0
                C = (1 + c1 * delta - c1 - cmu) * C \
                    + c1 * np.outer(pc, pc) \
                    + cmu * np.sum(w[:mu, None] * np.outer(ys[:mu], ys[:mu]), axis=0)
                
                # Ensure positive definite
                # Use eigenvalue decomposition or just add small identity if needed
                # For dim=5, Cholesky is cheap.
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    # If not PD, add identity
                    C = C + 1e-5 * np.eye(dim)
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                
                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Check convergence
                if np.linalg.norm(x_mean - x_mean_old) < 1e-8 * sigma:
                    break

        # Final polishing (optional, but CMA-ES usually handles it)
        # If we have remaining budget, do a few (1+1) steps
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            sigma_local = 0.1
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_new = np.clip(x_local + sigma_local * z, -5, 5)
                f_new = func(x_new)
                evals_used += 1
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9
                sigma_local = max(sigma_local, 1e-7)

        return best_x, best_f
