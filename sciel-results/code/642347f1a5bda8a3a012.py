import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 135.0  # Leave 15s buffer for safety (evaluator kills at 150s)
    
    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        Returns an array of correlation values for shifts k = -(N-1) ... -1, 1 ... N-1.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def objective_lse(h, beta, lambda_sum):
        """
        Objective: LSE(correlations) + penalty(sum - N/2)^2
        """
        g = 1.0 - h
        c_all = np.concatenate([np.correlate(h, g, mode='full')[:N-1], 
                                 np.correlate(h, g, mode='full')[N:]])
        
        # LogSumExp for numerical stability
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500
        exps = np.exp(beta * (c_all - max_c))
        sum_exps = np.sum(exps)
        lse = max_c + np.log(sum_exps) / beta
        
        # Sum penalty
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - N/2)**2
        
        return lse + sum_pen

    def objective_epigraph(h, t, lambda_sum):
        """
        Epigraph formulation: minimize t + penalty(sum - N/2)^2
        subject to C_k <= t for all k.
        We use a smooth approximation: sum_k max(0, C_k - t)^2 + t + penalty
        """
        g = 1.0 - h
        c_all = np.concatenate([np.correlate(h, g, mode='full')[:N-1], 
                                 np.correlate(h, g, mode='full')[N:]])
        
        # Smooth max: use soft plus for each violation
        violations = np.maximum(0.0, c_all - t)
        violation_penalty = np.sum(violations**2)
        
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - N/2)**2
        
        return t + violation_penalty + sum_pen

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = N/2.
        Uses a more robust approach.
        """
        h = np.array(h, dtype=float).copy()
        
        # Iteratively adjust to satisfy sum constraint while respecting bounds
        for _ in range(100):
            current_sum = np.sum(h)
            diff = N/2 - current_sum
            
            if abs(diff) < 1e-15:
                break
            
            # Determine which elements can be adjusted
            if diff > 0:
                # Need to increase sum: increase elements that are < 1
                mask = h < 1.0 - 1e-15
                if not np.any(mask):
                    break
                n_incr = np.sum(mask)
                step = diff / n_incr
                h[mask] = np.minimum(h[mask] + step, 1.0)
            else:
                # Need to decrease sum: decrease elements that are > 0
                mask = h > 1e-15
                if not np.any(mask):
                    break
                n_decr = np.sum(mask)
                step = diff / n_decr
                h[mask] = np.maximum(h[mask] + step, 0.0)
        
        # Final clip
        h = np.clip(h, 0.0, 1.0)
        
        # One more correction if needed
        current_sum = np.sum(h)
        if abs(current_sum - N/2) > 1e-12:
            # Distribute the remaining difference among elements not at bounds
            diff = N/2 - current_sum
            if diff > 0:
                mask = h < 1.0
            else:
                mask = h > 0.0
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        
        return h

    def optimize_lse(h0, beta, lambda_sum, max_iter, ftol=1e-9, gtol=1e-9):
        """Optimize using LSE surrogate."""
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            return objective_lse(h, beta, lambda_sum)

        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        # Project to feasible set
        h_opt = project_to_feasible(h_opt)
        
        return h_opt

    def optimize_epigraph(h0, t0, lambda_sum, max_iter, ftol=1e-9, gtol=1e-9):
        """Optimize using epigraph formulation with auxiliary variable t."""
        # Variables: [h_0, ..., h_{N-1}, t]
        # Bounds: h_i in [0,1], t in [0, N/2]
        bounds = [(0, 1) for _ in range(N)] + [(0, N/2)]
        
        def obj(x):
            h = x[:N]
            t = x[N]
            return objective_epigraph(h, t, lambda_sum)

        x0 = np.concatenate([h0, [t0]])
        
        try:
            res = scipy.optimize.minimize(
                obj, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            x_opt = res.x
            h_opt = x_opt[:N]
        except Exception:
            h_opt = h0.copy()
        
        # Project to feasible set
        h_opt = project_to_feasible(h_opt)
        
        return h_opt

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)

    rng = np.random.default_rng(42)
    
    # List of restart configurations: (h0, method, params, max_iter)
    # method: 'lse' or 'epigraph'
    restarts = []
    
    # 1. Uniform start with LSE
    restarts.append((np.full(N, 0.5), 'lse', (5.0, 100.0), 300))
    
    # 2-5. Random perturbations with LSE
    for i in range(4):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        beta = 5.0 + i * 5.0
        lambda_sum = 100.0 + i * 50.0
        
        restarts.append((h_rand, 'lse', (beta, lambda_sum), 300))
        
    # 6-9. Bimodal initializations with LSE
    for i in range(4):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.9, 0.1)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        beta = 15.0 + i * 5.0
        lambda_sum = 200.0 + i * 50.0
        
        restarts.append((h_rand, 'lse', (beta, lambda_sum), 300))
    
    # 10-13. Epigraph formulations with different t initializations
    for i in range(4):
        h_rand = rng.uniform(0.2, 0.8, size=N)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        t0 = score_h(h_rand) + 0.5  # Start with t slightly above current max
        lambda_sum = 200.0 + i * 50.0
        
        restarts.append((h_rand, 'epigraph', (t0, lambda_sum), 300))

    for idx, (h0, method, params, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit:
            break
            
        if method == 'lse':
            beta, lambda_sum = params
            h_opt = optimize_lse(h0, beta, lambda_sum, max_iter)
        else:  # epigraph
            t0, lambda_sum = params
            h_opt = optimize_epigraph(h0, t0, lambda_sum, max_iter)
        
        current_score = score_h(h_opt)
        
        if not np.isclose(np.sum(h_opt), N/2, atol=1e-8):
            continue
        if not (np.all(h_opt >= -1e-12) and np.all(h_opt <= 1 + 1e-12)):
            continue
            
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
    
    # Final polish pass with tighter tolerances
    if time.time() - start_time < time_limit - 15:
        # Use the best solution found so far as starting point
        h_polish = best_h.copy()
        
        # Try several high-beta polish iterations with LSE
        for beta_polish in [30.0, 50.0, 100.0]:
            if time.time() - start_time > time_limit:
                break
            
            h_opt = optimize_lse(h_polish, beta_polish, 500.0, 800, ftol=1e-11, gtol=1e-11)
            
            current_score = score_h(h_opt)
            
            if np.isclose(np.sum(h_opt), N/2, atol=1e-10) and \
               np.all(h_opt >= -1e-12) and np.all(h_opt <= 1 + 1e-12):
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
                    h_polish = h_opt.copy()
        
        # Try epigraph polish
        if time.time() - start_time < time_limit - 10:
            t0 = best_score + 0.1
            h_opt = optimize_epigraph(h_polish, t0, 1000.0, 800, ftol=1e-11, gtol=1e-11)
            
            current_score = score_h(h_opt)
            
            if np.isclose(np.sum(h_opt), N/2, atol=1e-10) and \
               np.all(h_opt >= -1e-12) and np.all(h_opt <= 1 + 1e-12):
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
    
    return best_h
