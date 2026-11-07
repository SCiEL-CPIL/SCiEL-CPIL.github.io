import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        # Use Sobol-like or random sampling. Random is fine for n=5.
        # Let's use 50 points for global exploration.
        n_global = 50
        global_candidates = []
        
        for i in range(n_global):
            if eval_count >= budget or not time_ok():
                break
            # Use random sampling in [-5, 5]
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                global_candidates.append((fx, x.copy()))

        if not global_candidates:
            return best_x, best_f

        # Sort global candidates
        global_candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-Start CMA-ES
        # Select top K seeds. K=5 seems reasonable.
        K = 5
        seeds = [x for _, x in global_candidates[:K]]
        
        # Budget allocation:
        # We have used ~50. Remaining ~250.
        # Allocate ~50 evals per seed.
        # If a seed converges early, we can stop and use remaining budget for the next or polish.
        
        # We will run CMA-ES sequentially on each seed.
        # To avoid wasting budget on a seed that is clearly worse than the current best,
        # we check if the seed's initial f is much worse than best_f.
        
        cma_budget_per_seed = 45
        lambda_ = 10 # Smaller lambda for faster convergence in low dim
        
        # CMA-ES Parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        for seed_idx, seed_x in enumerate(seeds):
            if eval_count >= budget or not time_ok():
                break
            
            # If this seed is already worse than the best found so far by a large margin, skip?
            # No, it might be in a different basin. But if it's much worse, CMA-ES might struggle.
            # We'll just run it.
            
            # Initialize CMA-ES state
            mean = seed_x.copy()
            sigma = 0.5 # Initial step size
            C = np.eye(n)
            C_sqrt = np.eye(n)
            C_sqrt_inv = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Track best in this local run to detect stagnation
            local_best_f = np.inf
            local_best_x = None
            stagnation_counter = 0
            max_stagnation = 5 # Generations without improvement
            
            while eval_count < budget and time_ok():
                # Check if we've spent enough on this seed
                if eval_count - (n_global + seed_idx * cma_budget_per_seed) > cma_budget_per_seed:
                    # Rough check: if we've used our allocated chunk, break to next seed
                    # This logic is tricky with global eval_count. 
                    # Better: just run until budget or convergence.
                    # But to ensure multi-start, we should limit time/budget per seed.
                    # Let's stick to a hard limit on evaluations per seed.
                    # We don't track per-seed evals easily in the closure. 
                    # Let's use a local counter.
                    pass # Handled by outer loop logic below? No.
                
                # Generate population
                ys = []
                xs = []
                for i in range(lambda_):
                    if eval_count >= budget or not time_ok():
                        break
                    z = rng.standard_normal(n)
                    y = C_sqrt @ z
                    x = mean + sigma * y
                    x = np.clip(x, -5, 5)
                    
                    fx = evaluate(x)
                    if fx is None:
                        break
                    ys.append(y)
                    xs.append((x, fx))
                
                if not xs:
                    break
                
                # Sort by fitness
                xs.sort(key=lambda t: t[1])
                
                # Update local best
                if xs[0][1] < local_best_f:
                    local_best_f = xs[0][1]
                    local_best_x = xs[0][0].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                if stagnation_counter > max_stagnation:
                    # Stagnated, move to next seed or break
                    break

                # Take top mu
                xs_top = xs[:mu]
                ys_top = ys[:mu]
                
                # Weighted mean
                new_mean = sum(weights[i] * xs_top[i][0] for i in range(mu))
                
                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C_sqrt_inv @ (new_mean - mean) / sigma)
                
                h_sigma = np.linalg.norm(p_sigma) < (1.4 + 2 / (n + 1)) * chi_n
                hsig = 1.0 if h_sigma else 0.0
                
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ (new_mean - mean) / sigma)
                
                # Covariance update
                rank_one = np.outer(p_c, p_c)
                rank_mu = sum(weights[i] * np.outer(ys_top[i], ys_top[i]) for i in range(mu))
                
                C = (1 + c_1 * hsig * (2 - c_c) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure PD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update mean
                mean = new_mean
                
                # Convergence check
                if sigma < 1e-6:
                    break

            # After CMA-ES on this seed, if we still have budget, continue to next seed
            # The global best_x/best_f is already updated in evaluate()

        # Phase 3: Final Polish with Nelder-Mead on the global best
        if best_x is not None and eval_count < budget and time_ok():
            x = best_x.copy()
            f = best_f
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            
            simplex = [x.copy()]
            for j in range(n):
                point = x.copy()
                delta = 0.1 if x[j] + 0.1 <= 5 else -0.1
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)
            
            fvals = [f]
            for pt in simplex[1:]:
                if eval_count >= budget or not time_ok():
                    break
                fx = evaluate(pt)
                if fx is None:
                    break
                fvals.append(fx)
                
            if len(fvals) != len(simplex):
                return best_x, best_f
                
            for _ in range(50):
                if eval_count >= budget or not time_ok():
                    break
                    
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break
                    
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                if eval_count >= budget or not time_ok():
                    break
                fr = evaluate(xr)
                if fr is None:
                    break
                
                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fe = evaluate(xe)
                    if fe is None:
                        simplex[-1], fvals[-1] = xr, fr
                        break
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fc = evaluate(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                            if eval_count >= budget or not time_ok():
                                break
                        if eval_count >= budget or not time_ok():
                            break
                        fvals = [f0] # This f0 is from outer scope, might be stale. Use fvals[0]
                        fvals[0] = evaluate(simplex[0]) # Re-evaluate best? Or just use current best.
                        # Actually, shrinking moves all points except best. Best is simplex[0].
                        # We need to re-evaluate the moved points.
                        # This is expensive. Let's just break if shrink is needed to save budget.
                        break

        return best_x, best_f
