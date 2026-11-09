"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with Rastrigin-aware initialization and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Rastrigin-aware global sampling
        # The global minimum of Rastrigin is at the center. With shift/rotation,
        # the minimum is at center + k (integer shifts). Sample a structured grid
        # plus random points to find a good starting basin.
        
        # First, sample a coarse grid to identify the approximate center
        n_grid = 5  # 5x5 grid in 5D is too many, so we do a sparse structured sample
        # Actually, let's just do a mix of structured and random
        # 10 random points + a few structured points near origin and boundaries
        
        initial_points = []
        # Origin and near-origin
        initial_points.append(np.zeros(dim))
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 0.5
                initial_points.append(p)
        
        # Random points
        while len(initial_points) < 20:
            initial_points.append(rng.uniform(-4.5, 4.5, dim))
        
        for p in initial_points:
            if call_count >= budget or time.time() > deadline:
                break
            val = call_func(np.clip(p, -5, 5))
            if val is None:
                break
        
        # Phase 2: Multi-start CMA-ES
        # Run CMA-ES from the best point found, and if stagnation occurs,
        # restart from other promising points or random locations.
        
        def run_cma_es(x0, sigma0, max_gens):
            nonlocal call_count, best_x, best_f, deadline
            
            x_mean = x0.copy()
            sigma = sigma0
            
            lambda_ = 12
            mu = lambda_ // 2
            
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            gen = 0
            stagnation = 0
            max_stagnation = 15
            
            while call_count < budget and time.time() < deadline and gen < max_gens and stagnation < max_stagnation:
                if gen % 10 == 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-12)
                        C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                    except:
                        C_inv_sqrt = np.eye(dim)
                
                offspring = []
                off_f = []
                
                for k in range(lambda_):
                    if call_count >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                    fy = call_func(y)
                    if fy is None:
                        break
                    off_f.append(fy)
                
                if len(off_f) < mu:
                    break
                
                sorted_idx = np.argsort(off_f)
                selected_x = [offspring[i] for i in sorted_idx[:mu]]
                selected_f = [off_f[i] for i in sorted_idx[:mu]]
                
                if selected_f[0] < best_f - 1e-10:
                    stagnation = 0
                else:
                    stagnation += 1
                
                m = sum(w * xk for w, xk in zip(weights, selected_x))
                
                diff = m - x_mean
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff
                
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) / chi_n
                if h_sigma < (1.4 + 2 / (dim + 1)):
                    sigma *= np.exp(c_sigma / chi_n * (h_sigma - 1))
                
                delta_d = (1 + c_mu * (mueff - 1)) * diff
                rank_mu = np.outer(delta_d, delta_d)
                p_c_norm_sq = np.linalg.norm(p_c)**2
                h_c = 1.0 if p_c_norm_sq > 1 / (1 - c_c)**2 else 0.0
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + h_c * np.eye(dim)) + c_mu * rank_mu
                
                x_mean = m
                gen += 1
                
                if sigma * np.max(np.sqrt(np.diag(C))) < 1e-6:
                    break
            
            return x_mean, sigma

        # Run CMA-ES from the best initial point
        run_cma_es(best_x.copy(), 1.5, 100)
        
        # If we still have budget, try a few more restarts from different points
        # This helps escape local minima in Rastrigin
        if call_count < budget - 20 and time.time() < deadline:
            # Try restarting from a few random points
            for restart in range(3):
                if call_count >= budget - 10 or time.time() > deadline:
                    break
                x_start = rng.uniform(-4, 4, dim)
                run_cma_es(x_start, 1.0, 50)
        
        # Phase 3: Fine-grained local refinement
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            for step in [0.1, 0.05, 0.02, 0.01, 0.005]:
                if call_count >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-12:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
