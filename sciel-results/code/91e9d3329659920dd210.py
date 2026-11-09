<<<<<<< SEARCH
"""
Strategy: Proper Diagonal CMA-ES with correct rank-1 and rank-mu updates on per-dimension
log-sigmas. The Discus function (f11) is highly anisotropic, so independent per-dimension
step sizes adapt rapidly to the ellipsoidal landscape. We use standard CMA-ES update
formulas adapted to the diagonal case, with careful budget allocation and a final
local polish phase.
"""

import numpy as np
import time
=======
"""
Strategy: Full CMA-ES with Cholesky-based covariance update for efficient handling of
the anisotropic Discus function. Uses standard CMA-ES with rank-1 and rank-mu updates
on the full covariance matrix, followed by a local refinement phase.
"""

import numpy as np
import time
>>>>>>> REPLACE
<<<<<<< SEARCH
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        best_calls_made = 0

        def call_f(x):
            nonlocal n_calls, best_x, best_f, best_calls_made
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                best_calls_made = n_calls
            return f_val

        def diag_cma_es(x0, sigma0_vec, max_calls):
            nonlocal n_calls, best_x, best_f, best_calls_made
            
            # Population size: lambda = 4 + 3*ln(n) for n=5 gives ~9, use 10 for stability
            lam = max(10, int(4 + 3 * np.log(dim)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            # CMA-ES parameters (standard)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_1 = (2 / ((dim + 1.7)**2 + mu_eff))
            c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff)) *
                       (c_sigma + 1 / d_sigma))
            chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            mean = x0.copy()
            log_sigmas = np.log(sigma0_vec.copy())  # Work in log-space for stability
            
            start_calls = n_calls
            stagnation_counter = 0
            max_stagnation = 3 * lam  # Allow more generations before declaring stagnation
            
            for gen in range(max_calls // lam + 20):
                if n_calls - start_calls >= max_calls:
                    break
                if time.time() > deadline - 3:
                    break
                
                # Generate population: x_i = mean + diag(exp(log_sigmas)) * z_i
                log_sigmas_clipped = np.clip(log_sigmas, -8, 8)
                sigmas = np.exp(log_sigmas_clipped)
                
                ys = rng.standard_normal((lam, dim))
                xs = mean + np.outer(np.ones(lam), sigmas) * ys
                xs = np.clip(xs, -5.0, 5.0)
                
                # Evaluate
                fs = np.zeros(lam)
                for i in range(lam):
                    f = call_f(xs[i])
                    if f is None:
                        return best_x, best_f
                    fs[i] = f
                
                # Sort by fitness
                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]
                ys = ys[idx]
                
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()
                    best_calls_made = n_calls
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter > max_stagnation:
                        break
                
                # Weighted recombination in y-space
                y_w = np.sum(w[:mu, np.newaxis] * ys[:mu], axis=0)
                
                # Update mean: mean_new = mean + diag(sigmas) * y_w
                mean = mean + sigmas * y_w
                mean = np.clip(mean, -5.0, 5.0)
                
                # Update log-sigmas using rank-1 and rank-mu updates
                # For diagonal covariance, the natural gradient updates for log(sigma_i) are:
                # Rank-1: c_1 * (y_w_i^2 - 1)
                # Rank-mu: c_mu * sum_k w_k * (y_{k,i}^2 - 1)
                # Combined: log(sigma_i) += 0.5 * (c_1 * (y_w_i^2 - 1) + c_mu * sum_k w_k * (y_{k,i}^2 - 1))
                # The factor 0.5 comes from the relationship between variance and log-sigma.
                
                rank_one = y_w ** 2 - 1.0
                rank_mu = np.sum(w[:mu, np.newaxis] * (ys[:mu] ** 2 - 1.0), axis=0)
                
                log_sigmas += 0.5 * (c_1 * rank_one + c_mu * rank_mu)
                
                # Clip to prevent numerical issues
                log_sigmas = np.clip(log_sigmas, -10, 10)
                
                # Check for convergence: if all sigmas are very small, we've converged
                if np.max(np.exp(log_sigmas)) < 1e-10:
                    break

            return best_x, best_f

        def local_polish(x_center, max_calls):
            """Simple coordinate descent / Nelder-Mead style local optimization"""
            nonlocal n_calls, best_x, best_f, best_calls_made
            
            x = x_center.copy()
            f_curr = call_f(x)
            if f_curr is None:
                return best_x, best_f
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()
                best_calls_made = n_calls
            
            # Use a simple adaptive step size local search
            step = np.full(dim, 0.1)
            no_improve = 0
            max_no_improve = 30
            
            while n_calls < budget - max_calls and time.time() < deadline - 5:
                if no_improve > max_no_improve:
                    break
                
                improved = False
                for d in range(dim):
                    for direction in [1.0, -1.0]:
                        x_trial = x.copy()
                        x_trial[d] += direction * step[d]
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        
                        f_trial = call_f(x_trial)
                        if f_trial is None:
                            return best_x, best_f
                        
                        if f_trial < f_curr:
                            f_curr = f_trial
                            x = x_trial
                            step[d] *= 1.5  # Expand step in this direction
                            improved = True
                            if f_curr < best_f:
                                best_f = f_curr
                                best_x = x.copy()
                                best_calls_made = n_calls
                            break
                    else:
                        # No improvement in either direction, shrink step
                        step[d] *= 0.5
                
                if not improved:
                    no_improve += 1
                    # Try a random perturbation if stuck
                    if no_improve > max_no_improve // 2:
                        x = x + rng.standard_normal(dim) * 0.05
                        x = np.clip(x, -5.0, 5.0)
                        f_curr = call_f(x)
                        if f_curr is None:
                            return best_x, best_f
                        if f_curr < best_f:
                            best_f = f_curr
                            best_x = x.copy()
                            best_calls_made = n_calls
                        no_improve = 0
                else:
                    no_improve = 0
            
            return best_x, best_f

        # Phase 1: Initial global search with Diagonal CMA-ES from center
        x0 = np.zeros(dim)
        sigma0_vec = np.full(dim, 2.5)  # Cover most of [-5,5]^5
        phase1_budget = int(budget * 0.6)  # 60% of budget for global search
        diag_cma_es(x0, sigma0_vec, phase1_budget)
        
        # Phase 2: Local polish around the best point found
        if n_calls < budget - 20 and time.time() < deadline - 10:
            local_polish(best_x, budget - n_calls)
        
        # Phase 3: If budget remains, try one more CMA-ES restart from best point
        remaining = budget - n_calls
        if remaining > 30 and time.time() < deadline - 10:
            x_restart = best_x + rng.standard_normal(dim) * 0.5
            x_restart = np.clip(x_restart, -5.0, 5.0)
            sigma_restart = np.full(dim, 0.5)
            diag_cma_es(x_restart, sigma_restart, remaining)
        
        return best_x, best_f
=======
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        best_calls_made = 0

        def call_f(x):
            nonlocal n_calls, best_x, best_f, best_calls_made
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                best_calls_made = n_calls
            return f_val

        def cma_es(x0, sigma0, max_calls):
            """Full CMA-ES with Cholesky-based covariance update."""
            nonlocal n_calls, best_x, best_f, best_calls_made
            
            # Population size
            lam = max(10, int(4 + 3 * np.log(dim)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            # CMA-ES parameters (standard)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_1 = (2 / ((dim + 1.7)**2 + mu_eff))
            c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff)) *
                       (c_sigma + 1 / d_sigma))
            chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)  # Covariance matrix
            pc = np.zeros(dim)  # Evolution path for covariance
            ps = np.zeros(dim)  # Evolution path for step size
            
            # Cholesky factor of C (initially identity)
            B = np.eye(dim)
            D = np.ones(dim)  # sqrt of eigenvalues
            
            start_calls = n_calls
            stagnation_counter = 0
            max_stagnation = 5 * lam
            
            for gen in range(max_calls // lam + 20):
                if n_calls - start_calls >= max_calls:
                    break
                if time.time() > deadline - 3:
                    break
                
                # Generate population
                zs = rng.standard_normal((lam, dim))
                # Transform: y = B * D * z
                ys = zs @ np.diag(D) @ B.T
                xs = np.tile(mean, (lam, 1)) + sigma * ys
                xs = np.clip(xs, -5.0, 5.0)
                
                # Evaluate
                fs = np.zeros(lam)
                for i in range(lam):
                    f = call_f(xs[i])
                    if f is None:
                        return best_x, best_f
                    fs[i] = f
                
                # Sort by fitness
                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]
                ys = ys[idx]
                
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()
                    best_calls_made = n_calls
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter > max_stagnation:
                        break
                
                # Weighted recombination
                y_w = np.sum(w[:mu, np.newaxis] * ys[:mu], axis=0)
                
                # Update mean
                mean = mean + sigma * y_w
                mean = np.clip(mean, -5.0, 5.0)
                
                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ y_w)) / sigma
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_N:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))
                
                # Update covariance matrix
                pc = (1 - c_1) * pc + np.sqrt(c_1 * (2 - c_1) * mu_eff) * y_w
                
                # Rank-1 update
                C = (1 + c_1 * (c_1 - 2 + c_mu) - c_mu) * C \
                    + c_1 * np.outer(pc, pc) \
                    - c_1 * (np.trace(C) / dim) * np.eye(dim)
                
                # Rank-mu update
                Hs = np.zeros((dim, dim))
                for i in range(mu):
                    Hs += w[i] * np.outer(ys[i], ys[i])
                C += c_mu * (Hs - C)
                
                # Recompute Cholesky factorization
                try:
                    # Add small regularization for numerical stability
                    C_reg = C + 1e-12 * np.eye(dim)
                    B_inv = np.linalg.cholesky(C_reg)
                    B = np.linalg.inv(B_inv)
                    D = np.sqrt(np.diag(B @ B.T))
                except np.linalg.LinAlgError:
                    # Fallback: use identity if Cholesky fails
                    B = np.eye(dim)
                    D = np.ones(dim)
                    C = np.eye(dim)
                
                # Check for convergence
                if sigma * np.min(D) < 1e-12:
                    break

            return best_x, best_f

        def local_polish(x_center, max_calls):
            """Nelder-Mead simplex method for local optimization"""
            nonlocal n_calls, best_x, best_f, best_calls_made
            
            # Initialize simplex
            x = x_center.copy()
            f_curr = call_f(x)
            if f_curr is None:
                return best_x, best_f
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()
                best_calls_made = n_calls
            
            # Create initial simplex
            simplex = [x.copy()]
            fs = [f_curr]
            for d in range(dim):
                x_new = x.copy()
                x_new[d] += 0.05
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = call_f(x_new)
                if f_new is None:
                    return best_x, best_f
                simplex.append(x_new)
                fs.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    best_calls_made = n_calls
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma_nm = 0.5  # Shrinkage
            
            max_iter = 100
            for it in range(max_iter):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                
                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = simplex[0].copy()
                    best_calls_made = n_calls
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    return best_x, best_f
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    best_calls_made = n_calls
                
                if f_r < fs[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        return best_x, best_f
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                        best_calls_made = n_calls
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fs[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                elif f_r < fs[-2]:
                    simplex[-1] = x_r
                    fs[-1] = f_r
                else:
                    # Contraction
                    if f_r < fs[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        return best_x, best_f
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                        best_calls_made = n_calls
                    if f_c < f_r:
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_s = call_f(simplex[i])
                            if f_s is None:
                                return best_x, best_f
                            fs[i] = f_s
                            if f_s < best_f:
                                best_f = f_s
                                best_x = simplex[i].copy()
                                best_calls_made = n_calls
                        break
                
                # Check convergence
                spread = np.max(np.array(fs) - np.min(np.array(fs)))
                if spread < 1e-10:
                    break
            
            return best_x, best_f

        # Phase 1: Global search with CMA-ES from center
        x0 = np.zeros(dim)
        sigma0 = 2.0  # Initial step size
        phase1_budget = int(budget * 0.5)  # 50% of budget for global search
        cma_es(x0, sigma0, phase1_budget)
        
        # Phase 2: Local polish around the best point found
        if n_calls < budget - 20 and time.time() < deadline - 10:
            local_polish(best_x, budget - n_calls)
        
        # Phase 3: If budget remains, try one more CMA-ES restart from best point
        remaining = budget - n_calls
        if remaining > 30 and time.time() < deadline - 10:
            x_restart = best_x + rng.standard_normal(dim) * 0.3
            x_restart = np.clip(x_restart, -5.0, 5.0)
            sigma_restart = 0.5
            cma_es(x_restart, sigma_restart, remaining)
        
        return best_x, best_f
>>>>>>> REPLACE
