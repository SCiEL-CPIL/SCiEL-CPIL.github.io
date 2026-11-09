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
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        call_count = 0
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return best_f
            call_count += 1
            try:
                # Clip to ensure we stay within bounds for the optimizer's internal state,
                # though func also clips.
                x = np.clip(x, -5, 5)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except:
                return 1e100

        rng = np.random.default_rng(42)
        
        # Strategy: 
        # 1. CMA-ES for global search and valley entry.
        #    CMA-ES adapts to the rotation and shift naturally.
        #    We use a small population to conserve budget but maintain diversity.
        # 2. Nelder-Mead for local polishing.
        #    NM is derivative-free and very robust for smooth functions like Rosenbrock
        #    once in the neighborhood of the optimum. It often converges faster than 
        #    FD-based L-BFGS-B in the final stages because it tracks the valley geometry 
        #    via simplex reflections/expansions.
        
        # CMA-ES Parameters
        pop_size = 12  # Standard for dim=5 is ~11, 12 is safe
        sigma = 2.5    # Initial step size, covering a good portion of [-5, 5]
        x_mean = np.zeros(dim)
        
        # CMA-ES State
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Precompute initial values
        # Inverse square root of identity is identity
        C_inv_sqrt = np.eye(dim)
        
        # Weights
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff) / (dim + 9)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Step size control state
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper to update CMA-ES state
        def cma_step():
            nonlocal x_mean, sigma, C, C_inv, C_sqrt, C_inv_sqrt, p_sigma, p_c
            nonlocal best_f, best_x, call_count
            
            if call_count + pop_size > budget:
                return False
            
            # Generate offspring
            ys = []
            xs = []
            fs = []
            
            # Use Cholesky decomposition of C for sampling
            # For dim=5, this is cheap.
            try:
                C_sqrt = np.linalg.cholesky(C)
            except:
                # Fallback to eigen if Cholesky fails (rare)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                C_inv = np.linalg.inv(C)
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                x = x_mean + sigma * y
                f = counted_func(x)
                ys.append(y)
                xs.append(x)
                fs.append(f)
                
            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update mean
            x_mean = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update sigma
            y_sigma = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            h_sigma = (np.linalg.norm(y_sigma) / np.sqrt(1 - (1 - sigma**2) ** (2 / pop_size))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff + 3))
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - np.sqrt(mu_eff)))
            # Corrected sigma update:
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - np.sqrt(mu_eff)))
            
            # Clip sigma
            sigma = np.clip(sigma, 1e-12, 100.0)
            
            # Update covariance
            y_c = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            h_c = (np.linalg.norm(y_c) / np.sqrt(1 - (1 - sigma**2) ** (2 / pop_size))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff + 3))
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            rank1 = np.outer(y_c, y_c)
            rankmu = np.sum([weights[i] * np.outer(ys[i], ys[i]) for i in range(mu)], axis=0)
            
            C = (1 + c_1 * (h_c * (1 + c_c) - c_c) + c_mu) * C + c_1 * rank1 + c_mu * rankmu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            C += 1e-12 * np.eye(dim)
            
            # Update inverses
            try:
                C_inv = np.linalg.inv(C)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
            except:
                pass
            
            return True

        # Phase 1: CMA-ES
        # Run until we have used ~60% of the budget or stagnate
        max_cma_calls = int(budget * 0.6)
        stagnation_count = 0
        prev_best = best_f
        
        while call_count < max_cma_calls and time.time() < deadline:
            if not cma_step():
                break
            
            # Check for stagnation
            if best_f < prev_best - 1e-8:
                prev_best = best_f
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # If stagnated for many iterations, break to polish
            if stagnation_count > 20:
                break
        
        # Phase 2: Nelder-Mead Polish
        # Use the best point found by CMA-ES as the starting point for NM.
        # NM is very effective for the final convergence.
        
        # We will run NM with a strict call limit to ensure we don't exceed budget.
        # NM uses dim+1 points. Each iteration uses ~1-2 calls.
        # We have ~40% budget left. Let's use a chunk of that.
        
        remaining_budget = budget - call_count
        # NM options
        # maxiter is number of iterations, not calls.
        # To be safe, we'll implement a custom NM loop to control calls precisely.
        
        def nelder_mead(func, x0, bounds, max_calls):
            nonlocal call_count, best_f, best_x
            
            dim = len(x0)
            # Initialize simplex
            simplex = []
            simplex_f = []
            
            # First point
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            simplex.append(x0)
            simplex_f.append(f0)
            
            # Other points: perturb by small amount
            base_step = 0.1
            for i in range(dim):
                xi = x0.copy()
                xi[i] += base_step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                simplex.append(xi)
                simplex_f.append(fi)
            
            # Sort
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while call_count < max_calls and time.time() < deadline:
                # Sort again just in case
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                
                if f_r < simplex_f[0]:
                    # Expand
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Replace
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contract
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
            
            # Find best in simplex
            best_idx = np.argmin(simplex_f)
            return simplex[best_idx], simplex_f[best_idx]

        # Run NM
        if call_count < budget - 10:
            try:
                nm_x, nm_f = nelder_mead(counted_func, best_x, None, budget)
                if nm_f < best_f:
                    best_f = nm_f
                    best_x = nm_x
            except Exception:
                pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
