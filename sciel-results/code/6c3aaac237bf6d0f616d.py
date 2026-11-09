import numpy as np
import time

class Optimizer:
    """
    Optimizer using Restarting CMA-ES for multimodal functions like Rastrigin.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # Safety margin
        
        rng = np.random.default_rng(42)
        
        # Global best tracking
        best_x = np.zeros(dim)
        best_fx = func(best_x)
        calls = 1
        
        # CMA-ES Parameters
        lam = 12
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Phase 1: Restarting CMA-ES
        # We allocate most of the budget here.
        max_phase1_calls = budget - 50  # Reserve 50 for polish
        
        x_best_global = best_x.copy()
        fx_best_global = best_fx
        
        while calls < max_phase1_calls and time.time() < deadline - 10:
            # Initialize or Re-initialize CMA-ES state
            # Start from the current global best, but with a fresh sigma
            # To encourage exploration, we can slightly perturb the start point
            start_x = x_best_global.copy()
            
            # Initial sigma: Start large to explore, then adapt
            # If we restart, we might want a larger sigma to jump
            sigma = 2.0 
            # On subsequent restarts, we could vary sigma, but 2.0 is a good default for [-5,5]
            
            C = np.eye(dim)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            stagnation_counter = 0
            best_in_run = fx_best_global
            
            # Run CMA-ES for this "run"
            while calls < max_phase1_calls and time.time() < deadline - 10:
                # Generate population
                X = np.zeros((lam, dim))
                F = np.zeros(lam)
                
                for i in range(lam):
                    if calls >= max_phase1_calls:
                        break
                    step = sigma * (C_sqrt @ rng.standard_normal(dim))
                    x_new = np.clip(start_x + step, -5, 5)
                    X[i] = x_new
                    F[i] = func(x_new)
                    calls += 1
                    
                    if F[i] < fx_best_global:
                        fx_best_global = F[i]
                        x_best_global = X[i].copy()
                    if F[i] < best_in_run:
                        best_in_run = F[i]
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

                if calls >= max_phase1_calls:
                    break
                    
                # Sort
                idx = np.argsort(F)
                X_sorted = X[idx]
                F_sorted = F[idx]
                
                # Update mean
                x_old = start_x.copy()
                start_x = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
                start_x = np.clip(start_x, -5, 5)
                
                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (start_x - x_old))
                norm_ps = np.linalg.norm(p_sigma)
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)
                
                # Update C
                h_sigma = norm_ps / np.sqrt(1 - (1 - c_sigma) ** 10) < (1 + 2 / (dim + 1)) * chi_n
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (start_x - x_old) / sigma
                
                C += c_1 * (np.outer(p_c, p_c) / sigma ** 2 - 1) * C
                y = (X_sorted[:mu] - x_old) / sigma
                C += c_mu * np.sum(weights[:, None, None] * np.outer(y[:, :, None], y[:, None, :]), axis=0)
                
                # Periodic re-eigen
                if calls % 50 < lam:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                    C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
                
                # Stagnation check: If we haven't improved the best in this run for many generations
                # and sigma is small, it's likely stuck in a local minimum.
                if stagnation_counter > 15 * lam and sigma < 0.5:
                    # Restart with a larger sigma to jump
                    break
                
                # Also break if sigma becomes extremely small (converged)
                if sigma < 1e-6:
                    break

        # Phase 2: Local Polish (Nelder-Mead or (1+1)-ES)
        # Use remaining budget
        if calls < budget and time.time() < deadline - 5:
            x_nm = x_best_global.copy()
            simplex = [x_nm.copy()]
            # Initial simplex size
            step_size = 0.1
            for i in range(dim):
                pert = x_nm.copy()
                pert[i] += step_size
                simplex.append(np.clip(pert, -5, 5))
            
            fvals = [func(s) for s in simplex]
            calls += dim + 1
            
            # Update best
            for i in range(dim + 1):
                if fvals[i] < fx_best_global:
                    fx_best_global = fvals[i]
                    x_best_global = simplex[i].copy()
            
            # Nelder-Mead iterations
            while calls < budget and time.time() < deadline - 5:
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = np.clip(2 * centroid - simplex[-1], -5, 5)
                fr = func(xr)
                calls += 1
                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + 2 * (xr - centroid), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            
                # Update best
                if fvals[0] < fx_best_global:
                    fx_best_global = fvals[0]
                    x_best_global = simplex[0].copy()

        return x_best_global, fx_best_global
