"""Optimizer for BBOB-style f17 (Schaffers F7) using a direct grid + CMA-ES approach."""
import numpy as np
import time


class Optimizer:
    """Direct approach: coarse grid search followed by CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_left():
            return deadline - time.time()

        # Phase 1: Latin Hypercube Sampling for better initial coverage
        n_initial = 100
        # Generate LHS points
        # For each dimension, divide [0,1] into n_initial bins, sample one point per bin, then permute
        lhs_points = np.zeros((n_initial, dim))
        for d in range(dim):
            bins = np.linspace(0, 1, n_initial + 1)
            points = rng.uniform(bins[:-1], bins[1:])
            rng.shuffle(points)
            lhs_points[:, d] = points
        
        # Scale to [-5, 5]
        lhs_points = lhs_points * 10 - 5
        
        for i in range(n_initial):
            if calls_used >= budget or time_left() < 5.0:
                break
            x = lhs_points[i]
            f = func(x)
            calls_used += 1
            update_best(x, f)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: Run CMA-ES from the best point found
        x_start = best_x.copy()
        
        # Adaptive initial sigma based on problem scale
        sigma_init = 2.0
        
        def run_cmaes(x_start, sigma_init, max_calls, min_sigma=1e-6):
            nonlocal calls_used, best_x, best_f
            
            # CMA-ES parameters
            lam = 20  # Population size for 5D
            mu = lam // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            # Step size adaptation
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (2 + mueff / dim) / (dim + cc + 2)
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + cc
            
            # Covariance matrix adaptation
            c1 = (mueff - 2 + 1 / dim) / ((dim + 2) ** 2 - mueff)
            cmu = min(1 - cc, 2 / ((dim + 1.3) ** 2 + mueff))
            cmu = min(cmu, (1 - 2 * c1) / 2)
            
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # State variables
            x_mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_calls = 0
            no_improve = 0
            max_no_improve = 40
            
            while local_calls < max_calls:
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                # 1. Sample new points
                ys = rng.standard_normal((lam, dim))
                xs = np.array([x_mean + sigma * (ys[i] @ C_sqrt) for i in range(lam)])
                # Clip to box
                xs = np.clip(xs, -5, 5)
                
                # Evaluate
                fs = np.zeros(lam)
                for i in range(lam):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    fs[i] = func(xs[i])
                    calls_used += 1
                    local_calls += 1
                    update_best(xs[i], fs[i])
                
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                
                # 2. Sort
                order = np.argsort(fs)
                xs = xs[order]
                fs = fs[order]
                ys = ys[order]
                
                # Check for improvement
                if fs[0] < best_f - 1e-12:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > max_no_improve:  # Stagnation check
                        break

                # 3. Update mean
                x_mean_new = np.sum(weights[:, None] * xs[:mu], axis=0)
                
                # 4. Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt.T @ ((x_mean_new - x_mean) / sigma))
                
                # Update sigma
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                
                # Update pc
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma
                
                # Update covariance matrix C
                ys_parents = ys[:mu]
                S_w = np.sum(weights[:, None, None] * (ys_parents[:, :, None] * ys_parents[:, None, :]), axis=0)
                
                C_new = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * C) + cmu * S_w
                
                # Ensure C is symmetric
                C_new = (C_new + C_new.T) / 2.0
                
                # Update C_sqrt (Cholesky decomposition)
                try:
                    C_sqrt = np.linalg.cholesky(C_new)
                except np.linalg.LinAlgError:
                    C_new += 1e-9 * np.eye(dim)
                    try:
                        C_sqrt = np.linalg.cholesky(C_new)
                    except:
                        C_sqrt = np.eye(dim)
                        break
                    
                C = C_new
                x_mean = x_mean_new

            return local_calls

        # Run CMA-ES with remaining budget
        remaining_budget = budget - calls_used
        run_cmaes(x_start, sigma_init, remaining_budget)

        # Phase 3: Fine-tuning with local search if time permits
        if best_x is not None and time_left() > 10.0 and calls_used < budget:
            # Use a short CMA-ES run with very small sigma for polishing
            x_polish_start = best_x.copy()
            sigma_polish = 0.1
            run_cmaes(x_polish_start, sigma_polish, min(100, budget - calls_used))

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_used < budget:
                best_f = func(best_x)
                calls_used += 1

        return best_x, best_f
