import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # 1. Global Exploration: Halton sequence for space-filling
        n_explore = 15
        rng = np.random.default_rng(42)
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim, start=1):
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                for i in range(n):
                    points[i, d] = halton(start + i, base)
            return points

        # Generate initial points
        h_points = halton_sequence(n_explore, dim)
        h_points = h_points * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        candidates = []

        for i in range(n_explore):
            if not time_ok():
                break
            x = h_points[i]
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Select 3 distinct starting points for parallel CMA-ES
        candidates.sort(key=lambda t: t[0])
        # Use top 3, but ensure they are somewhat distinct to avoid redundant work
        # Simple check: if too close, pick next best
        starts = []
        for fx, x in candidates:
            if len(starts) == 3:
                break
            is_distinct = True
            for sx in starts:
                if np.linalg.norm(x - sx) < 0.5:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append(x)
        
        # If we don't have 3 distinct ones, just take the top 3 available
        if len(starts) < 3:
            for fx, x in candidates:
                if x not in starts:
                    starts.append(x)
                if len(starts) == 3:
                    break

        # 2. Parallel CMA-ES
        # We split the remaining budget roughly equally among the 3 runs.
        # CMA-ES is population-based, so it uses more calls per iteration than (1+1)-ES,
        # but it converges faster in high dimensions and handles multimodality better via restarts/parallelism.
        
        remaining_budget = budget - n_explore
        if remaining_budget <= 0 or not starts:
            return best_x, best_f

        # Split budget: 40%, 30%, 30% roughly, adjusted by time
        # CMA-ES lambda is typically 4 + floor(3 ln N) = 4 + 3*1.6 = 8 for N=5
        # Let's use lambda=6 for efficiency in 5D.
        lambda_cma = 6
        
        # We will run 3 CMA-ES instances.
        # To manage the global budget strictly, we will track a global call counter.
        global_calls = n_explore
        global_best_f = best_f
        global_best_x = best_x

        def cma_es(x0, sigma0, max_calls, global_state):
            # local state
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            Cs = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Parameters
            cmu = 1.0 / (4 * dim + dim**0.5 + 2)
            c_sigma = (mu_eff + 2) / (dim + 4)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(cmu, (2 - c_1 + cmu) / ((dim + 2 + c_1)**2 + mu_eff))
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # mu is floor(lambda/2)
            mu = lambda_cma // 2
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(i + 1) - np.log(mu)
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            # Recompute rates with correct mu_eff
            c_sigma = (mu_eff + 2) / (dim + 4)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(cmu, (2 - c_1 + cmu) / ((dim + 2 + c_1)**2 + mu_eff))
            
            calls_used = 0
            local_best_f = np.inf
            local_best_x = x0.copy()

            # Initial evaluation of mean (optional, but good for baseline)
            # Actually, CMA-ES usually starts by sampling. Let's just start the loop.
            
            while calls_used < max_calls and global_state['calls'] < budget and time_ok():
                # Check if we have enough budget for a full population
                if global_state['calls'] + lambda_cma > budget:
                    # Do a partial step or stop
                    break
                
                # Sample population
                Z = rng.multivariate_normal(np.zeros(dim), C, size=lambda_cma)
                Y = x_mean + sigma * Z
                Y = np.clip(Y, -5, 5)
                
                # Evaluate population
                F = np.zeros(lambda_cma)
                for i in range(lambda_cma):
                    if global_state['calls'] >= budget or not time_ok():
                        # Fill rest with inf if we run out
                        F[i] = np.inf
                        continue
                    F[i] = func(Y[i])
                    global_state['calls'] += 1
                    calls_used += 1
                    if F[i] < local_best_f:
                        local_best_f = F[i]
                        local_best_x = Y[i].copy()
                    if F[i] < global_state['best_f']:
                        global_state['best_f'] = F[i]
                        global_state['best_x'] = Y[i].copy()

                if calls_used < lambda_cma:
                    break

                # Sort by fitness
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]
                
                # Update mean
                x_new = np.sum(w[:, np.newaxis] * Y[:mu], axis=0)
                
                # Update covariance matrix
                y_vec = (x_new - x_mean) / sigma
                Cs = (1 - c_sigma) * Cs + c_sigma * (2 - c_sigma) * np.outer(y_vec, y_vec)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((global_state['iter']+1)/(dim/c_sigma))) < (1.4 + 2/(dim+1)) * chi_N)
                if hsig:
                    pc = (1 - c_c) * pc + c_c * y_vec
                
                # Damping
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((global_state['iter']+1)/(dim/c_sigma))) < (1.4 + 2/(dim+1)) * chi_N)
                
                C = (1 - c_1 - c_mu + (not hsig) * c_1 * c_c * (2 + c_c) * (2 + c_c) - c_mu) * C
                C += c_1 * (np.outer(pc, pc) - C)
                if hsig:
                    C += c_mu * (np.outer(y_vec, y_vec) - C)
                else:
                    # Standard rank-mu update
                    C += c_mu * np.sum(w[:, np.newaxis] * np.outer(Y[:mu] - x_mean, Y[:mu] - x_mean) / (sigma**2), axis=0)
                
                # Symmetrize and fix eigenvalues
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(ps) / chi_N - 1))
                
                # Update mean
                x_mean = x_new
                global_state['iter'] += 1
                
                # Safety: if sigma becomes too small or large
                sigma = np.clip(sigma, 1e-8, 10.0)

            return local_best_f, local_best_x

        # Initialize global state
        g_state = {
            'calls': global_calls,
            'best_f': global_best_f,
            'best_x': global_best_x.copy(),
            'iter': 0
        }

        # Run 3 CMA-ES instances
        # Allocate budget: 40%, 30%, 30%
        alloc = [0.4, 0.3, 0.3]
        for i in range(3):
            if not time_ok():
                break
            if i >= len(starts):
                break
            
            max_calls_for_this = int(remaining_budget * alloc[i])
            if max_calls_for_this < 10:
                continue
                
            x0 = starts[i]
            # Initial sigma: slightly smaller than box size
            sigma0 = 1.0
            
            # Run CMA-ES
            try:
                f_res, x_res = cma_es(x0, sigma0, max_calls_for_this, g_state)
            except Exception:
                # If CMA-ES crashes, fallback to simple local search for remaining budget
                pass

            # Update global best
            if g_state['best_f'] < global_best_f:
                global_best_f = g_state['best_f']
                global_best_x = g_state['best_x'].copy()

        # 3. Final Polish: (1+1)-ES on the best found point with remaining budget
        if g_state['calls'] < budget and time_ok():
            remaining = budget - g_state['calls']
            if remaining > 5:
                x_cur = global_best_x.copy()
                f_cur = global_best_f
                sigma = 0.1
                
                for _ in range(remaining):
                    if not time_ok() or g_state['calls'] >= budget:
                        break
                    z = rng.standard_normal(dim)
                    y = x_cur + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    g_state['calls'] += 1
                    
                    if fy < f_cur:
                        f_cur = fy
                        x_cur = y.copy()
                        sigma *= 1.19
                    else:
                        sigma *= 0.84
                    
                    sigma = np.clip(sigma, 1e-8, 5.0)
                    
                    if fy < global_best_f:
                        global_best_f = fy
                        global_best_x = x_cur.copy()

        return global_best_x, global_best_f
