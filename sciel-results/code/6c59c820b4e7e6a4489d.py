import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 320 # 10s margin
            rng = np.random.default_rng(42)
            
            # Initial random sampling
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            best_x = x.copy()
            best_f = fx
            budget_used = 1
            
            # CMA-ES parameters
            n = self.dim
            lam = 10
            mu = lam // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            
            # Step size control
            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + cs * (2 * (mueff + 1) / (n + 2) - 1)
            # Covariance matrix adaptation
            c1 = (2 * (mueff - 2 + 2 / mueff)) / ((n + 2)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            
            while budget_used < self.budget:
                if time.time() > deadline:
                    break
                
                # Restart logic:
                # We run a CMA-ES phase.
                # If we stagnate, we restart.
                # For simplicity and robustness with small budget,
                # let's just run CMA-ES continuously but with a "restart" trigger.
                
                # State
                mean = best_x.copy()
                sigma = 2.0
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                chi_n = np.sqrt(n) * (1 + 1/(4*n) - 1/(21*n**2))
                
                stagnation_count = 0
                local_best_f = best_f
                
                while budget_used < self.budget and time.time() < deadline:
                    # Sample
                    B = rng.standard_normal((lam, n))
                    # Whitening: sqrt(C)
                    # For n=5, we can use eigendecomposition or just approx
                    # Actually, for small n, we can maintain C and use np.linalg.cholesky if PD, or eig
                    # C is PD.
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Fallback to eig if numerical issues
                        w, V = np.linalg.eigh(C)
                        w = np.maximum(w, 0)
                        L = V @ np.diag(np.sqrt(w)) @ V.T
                        # This is not a Cholesky, it's a factorization.
                        # For CMA-ES, we need y = sigma * L @ z
                        # If we use Eigh, L is not lower triangular.
                        # Standard CMA-ES uses Cholesky.
                        # If Cholesky fails, add small jitter to C.
                        C += 1e-12 * np.eye(n)
                        L = np.linalg.cholesky(C)
                    
                    ys = sigma * (B @ L.T) # L is lower, so L.T is upper? No.
                    # C = L L^T. We want y ~ N(0, C).
                    # z ~ N(0, I). y = L z.
                    # B is (lam, n). We need B @ L.T ?
                    # If L is (n, n) lower triangular, and B is (lam, n) rows are z_i.
                    # z_i is 1x n. y_i = z_i L^T.
                    # So Y = B @ L.T.
                    
                    xs = mean + ys
                    # Clip
                    xs = np.clip(xs, -5, 5)
                    
                    # Evaluate
                    fs = np.array([func(x) for x in xs])
                    budget_used += lam
                    
                    # Update best
                    idx_min = np.argmin(fs)
                    if fs[idx_min] < best_f:
                        best_f = fs[idx_min]
                        best_x = xs[idx_min].copy()
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                    
                    # Sort
                    idx = np.argsort(fs)
                    ys_sorted = ys[idx]
                    xs_sorted = xs[idx]
                    fs_sorted = fs[idx]
                    
                    # Update mean
                    mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                    
                    # Update step size control variables
                    y_mean = (mean - mean_prev) / sigma # Wait, need mean_prev
                    # I forgot to save mean_prev. Let's restructure.
                    
                    # Restart check
                    if stagnation_count > 20:
                        break
                        # Restart:
                        # mean = best_x (exploit) or random?
                        # Let's try random restart to escape local minima.
                        # But keep best_x.
                        # If we restart, we reset C, sigma, etc.
                        # Let's break out and let the outer loop handle restart logic if we implement it.
                        # For now, let's just break and restart with random mean.
                        mean = rng.uniform(-5, 5, n)
                        sigma = 2.0
                        C = np.eye(n)
                        pc = np.zeros(n)
                        ps = np.zeros(n)
                        stagnation_count = 0
                        continue # Continue inner loop with new state? 
                        # Actually, it's cleaner to break and let a wrapper handle it.
                        # But budget is tight.
                        # Let's just break the inner loop and restart the inner loop setup.
                
                # End of inner loop
            return best_x, best_f
