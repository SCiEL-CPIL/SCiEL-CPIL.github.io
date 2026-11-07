# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Coarse global exploration via stratified sampling ---
        # We want good initial coverage. Use a grid of 3^d points? 3^5=243.
        # Or random points. Let's do a mix: center + random.
        # Actually, for Katsuura with hidden rotation, the minimizer is shifted.
        # A dense random sample is safer than a grid aligned with axes.
        
        # Let's use a Latin Hypercube or just random points.
        # Budget is 500. Let's spend ~100 on global exploration.
        n_global = 100
        for i in range(n_global):
            if check_time() or used_calls >= budget:
                break
            x = rng.uniform(-5, 5, d)
            f = func(x)
            used_calls += 1
            try_update(x, f)

        # --- Phase 2: CMA-ES like local refinement ---
        # We will run a few instances of a simple CMA-ES variant.
        # Since we don't have pyCMA, we implement a basic (mu+lambda)-ES or (1+1)-ES with covariance adaptation.
        # A simple robust approach: Nelder-Mead is okay but can be slow.
        # Let's implement a basic CMA-ES from scratch.
        
        def cma_es(x0, f0, budget_left, sigma0=0.5):
            nonlocal used_calls, best_f, best_x
            if budget_left <= 0 or check_time():
                return
            
            # Initialize
            x = x0.copy()
            f = f0
            sigma = sigma0
            C = np.eye(d)
            C_sqrt = np.eye(d)
            pc = np.zeros(d)
            cs = 1.0
            c1 = 1 / (d + 4)
            cmu = 1 / (d + 8)
            damps = 1 + 2 * max(0, np.sqrt((d - 1) / (d + 3)) - 1) + c1
            cc = (4 + d / 4) / (d + 4 + 2 * d / 4)
            cs = (4 + d / 4) / (d + 4 + 2 * d / 4)
            chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            
            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(d)))
            mu = lambda_ // 2
            
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)
            
            # Step size adaptation
            while used_calls < budget and not check_time():
                # Generate population
                pop = []
                for i in range(lambda_):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    y = C_sqrt @ z
                    xi = x + sigma * y
                    fi = func(xi)
                    used_calls += 1
                    pop.append((xi, fi))
                
                if len(pop) < lambda_:
                    break
                
                # Sort
                pop.sort(key=lambda t: t[1])
                
                # Update best
                if pop[0][1] < best_f:
                    best_f = pop[0][1]
                    best_x = pop[0][0].copy()
                
                # Compute mean of top mu
                x_new = np.zeros(d)
                for i in range(min(mu, len(pop))):
                    x_new += weights[i] * pop[i][0]
                
                # Update step size
                y_mean = (x_new - x) / sigma if sigma > 0 else np.zeros(d)
                cs = (1 - cs) * cs + cs * (np.linalg.norm(y_mean) / chiN - 1)
                sigma *= np.exp(cs / damps * (np.linalg.norm(y_mean) / chiN - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)
                
                # Update path
                pc = (1 - cs) * pc + cs * y_mean
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs)**2) / chiN < 1.4 + 2 / (d + 1))
                
                # Update covariance
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (1 - cs) * C)
                for i in range(min(mu, len(pop))):
                    yi = (pop[i][0] - x) / sigma
                    C += cmu * weights[i] * np.outer(yi, yi)
                
                x = x_new
                
                # Recompute C_sqrt periodically (expensive)
                if used_calls % (10 * lambda_) == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                
                # Check for stagnation or convergence
                if sigma < 1e-8:
                    break
                if np.all(np.abs(x_new - x) < 1e-8):
                    break

        # Run CMA-ES from the best point found in global phase
        if best_x is not None:
            cma_es(best_x, best_f, budget - used_calls, sigma0=1.0)

        # --- Phase 3: Local polish with Nelder-Mead or simple gradient-free ---
        # If we have budget left, do a fine local search.
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            f = best_f
            # Simple (1+1)-ES with very small steps
            sigma = 0.01
            for _ in range(budget - used_calls):
                if check_time() or used_calls >= budget:
                    break
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < f:
                    x = y
                    f = fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                sigma = float(np.clip(sigma, 1e-6, 5.0))
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
