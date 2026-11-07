"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Single-trajectory Rank-One CMA-ES with Nelder-Mead polish for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() - start_time > max_time:
                return float('inf')
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Brief exploration to find a good starting basin
        # Use a small set of diverse starting points
        n_explore = min(20, budget // 10)
        explore_points = []
        explore_points.append(np.zeros(dim))
        for _ in range(n_explore - 1):
            explore_points.append(rng.uniform(-4.5, 4.5, dim))
        
        explore_results = []
        for x in explore_points:
            if calls >= budget or time.time() - start_time > max_time:
                break
            fx = safe_call(x)
            if fx != float('inf'):
                explore_results.append((fx, x.copy()))
        
        explore_results.sort(key=lambda t: t[0])
        
        # Pick the best exploration point as starting point for CMA-ES
        if explore_results:
            start_mean = explore_results[0][1].copy()
        else:
            start_mean = np.zeros(dim)

        # Phase 2: Rank-One CMA-ES (single trajectory with adaptive covariance)
        # This is more efficient than population-based ES for low-dim problems
        # and adapts to the skewed geometry of Lunacek's basin
        
        # CMA-ES parameters
        sigma = 1.5
        C = np.eye(dim)  # Covariance matrix
        c_p = 2 / (dim + 4)  # Path length adaptation rate
        c_c = (4 + dim / dim) / (dim + 4 + 3)  # Hessian adaptation rate
        c_1 = 2 / ((dim + 1.3)**2 + dim)  # Covariance adaptation rate (rank-1)
        c_mu = min(1 - c_1, 2 / ((dim + 1)**2 + dim))  # Cumulative step size adaptation
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        count_success = 0
        no_improve = 0
        max_no_improve = 100
        
        # Initial population size for CMA-ES (we'll use a small effective population)
        # For rank-one update, we sample a few points but only use the best for the update
        n_sample = 4  # Number of samples per iteration
        
        while calls < budget and time.time() - start_time < max_time:
            # Check time budget
            if time.time() - start_time > max_time:
                break
            
            # Generate samples from N(mean, sigma^2 * C)
            samples = []
            for _ in range(n_sample):
                z = rng.standard_normal(dim)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                x_sample = start_mean + sigma * (L @ z)
                x_sample = np.clip(x_sample, -5, 5)
                fx = safe_call(x_sample)
                if fx != float('inf'):
                    samples.append((fx, x_sample.copy()))
            
            if not samples:
                break
            
            # Sort samples by fitness
            samples.sort(key=lambda t: t[0])
            best_sample_f, best_sample_x = samples[0]
            
            # Update best if improved
            if best_sample_f < best_f:
                best_f = best_sample_f
                best_x = best_sample_x.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Rank-one update based on the best sample
            # Compute the step
            step = best_sample_x - start_mean
            step_norm = np.linalg.norm(step)
            
            if step_norm > 1e-12:
                # Update path adaptation
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (step / (sigma * step_norm))
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (step / (sigma * step_norm))
                
                # Update covariance (rank-one update)
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1-c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_n else 0
                C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1) * C + c_1 * np.outer(step/sigma, step/sigma)
                
                # Update mean
                start_mean = best_sample_x.copy()
                
                # Adapt sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma)/chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 5.0)
            
            # Stagnation restart: if no improvement for a while, restart around best
            if no_improve >= max_no_improve:
                # Reset CMA-ES state around current best
                start_mean = best_x.copy()
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 0.8
                no_improve = 0
                max_no_improve = 50  # Reduce threshold for subsequent restarts
                max_no_improve = max(max_no_improve, 20)

        # Phase 3: Nelder-Mead polishing on the best solution
        if calls < budget and time.time() - start_time < max_time - 5:
            nm_budget = budget - calls
            if nm_budget > 10:
                # Build initial simplex around best_x
                simplex = [best_x.copy()]
                base_sigma = 0.3
                for i in range(dim):
                    pt = best_x.copy()
                    pt[i] += base_sigma
                    pt = np.clip(pt, -5, 5)
                    simplex.append(pt)
                
                # Evaluate simplex
                simplex_f = []
                for pt in simplex:
                    if calls >= budget or time.time() - start_time > max_time:
                        break
                    f = safe_call(pt)
                    simplex_f.append(f)
                
                if len(simplex_f) != dim + 1:
                    # Not enough budget to complete simplex initialization
                    # Fall back to simple hill climbing
                    if calls < budget and time.time() - start_time < max_time:
                        for _ in range(30):
                            if calls >= budget or time.time() - start_time > max_time:
                                break
                            direction = rng.standard_normal(dim)
                            candidate = best_x + 0.1 * direction
                            candidate = np.clip(candidate, -5, 5)
                            fc = safe_call(candidate)
                            if fc != float('inf') and fc < best_f:
                                best_f = fc
                                best_x = candidate.copy()
                else:
                    # Nelder-Mead parameters
                    alpha = 1.0
                    gamma = 2.0
                    rho = 0.5
                    sigma_shrink = 0.5
                    
                    nm_iter = 0
                    max_nm_iter = min(nm_budget - 1, 80)
                    
                    while nm_iter < max_nm_iter and calls < budget and time.time() - start_time < max_time:
                        nm_iter += 1
                        
                        # Sort simplex by fitness
                        order = np.argsort(simplex_f)
                        simplex = [simplex[i] for i in order]
                        simplex_f = [simplex_f[i] for i in order]
                        
                        # Check convergence
                        if simplex_f[-1] - simplex_f[0] < 1e-10:
                            break
                        
                        # Centroid of all but worst point
                        centroid = np.mean(simplex[:-1], axis=0)
                        
                        # Reflection
                        worst_pt = simplex[-1]
                        worst_f = simplex_f[-1]
                        x_ref = centroid + alpha * (centroid - worst_pt)
                        x_ref = np.clip(x_ref, -5, 5)
                        
                        if calls >= budget or time.time() - start_time > max_time:
                            break
                        f_ref = safe_call(x_ref)
                        
                        if f_ref < simplex_f[0]:
                            # Expansion
                            x_exp = centroid + gamma * (x_ref - centroid)
                            x_exp = np.clip(x_exp, -5, 5)
                            if calls >= budget or time.time() - start_time > max_time:
                                break
                            f_exp = safe_call(x_exp)
                            if f_exp < f_ref:
                                x_new, f_new = x_exp, f_exp
                            else:
                                x_new, f_new = x_ref, f_ref
                            simplex[-1] = x_new
                            simplex_f[-1] = f_new
                            if f_new < best_f:
                                best_f = f_new
                                best_x = x_new.copy()
                        elif f_ref < simplex_f[-2]:
                            # Accept reflection
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                            if f_ref < best_f:
                                best_f = f_ref
                                best_x = x_ref.copy()
                        else:
                            # Contraction
                            x_con = centroid + rho * (worst_pt - centroid)
                            x_con = np.clip(x_con, -5, 5)
                            if calls >= budget or time.time() - start_time > max_time:
                                break
                            f_con = safe_call(x_con)
                            if f_con < worst_f:
                                simplex[-1] = x_con
                                simplex_f[-1] = f_con
                                if f_con < best_f:
                                    best_f = f_con
                                    best_x = x_con.copy()
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], -5, 5)
                                    if calls >= budget or time.time() - start_time > max_time:
                                        break
                                    f = safe_call(simplex[i])
                                    simplex_f[i] = f
                                    if f < best_f:
                                        best_f = f
                                        best_x = simplex[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
