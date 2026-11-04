import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 138.0  # Leave 12s buffer for safety (evaluator kills at 150s)
    
    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        Returns an array of correlation values for shifts k = -(N-1) ... -1, 1 ... N-1.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def correlations_and_grad(h):
        """
        Compute correlations C_k and their gradients dC_k/dh_i.
        Returns (c_all, grad) where grad has shape (N, len(c_all)).
        """
        g = 1.0 - h
        c_full = np.correlate(h, g, mode='full')
        c_all = np.concatenate([c_full[:N-1], c_full[N:]])
        
        # C_k = sum_i h[i] * g[i+k] for shift k
        # dC_k/dh_j = g[j+k] - h[j] (since g = 1-h, dg/dh = -1)
        # Actually: C_k = sum_i h[i] * (1-h[i+k])
        # dC_k/dh_j = (1-h[j+k]) + sum_i h[i] * (-delta_{i+k,j})
        # = (1-h[j+k]) - h[j-k]  (with appropriate bounds)
        
        # More carefully using the correlation definition:
        # np.correlate(h, g, mode='full')[m] = sum_i h[i+m-N+1] * g[i] for valid i
        # This is getting complex. Let's use a simpler approach:
        # C_k = sum_i h[i] * (1-h[(i+k) mod N]) for circular, but we have linear.
        
        # For linear correlation:
        # c_full[m] where m goes from 0 to 2N-2
        # c_full[m] = sum_i h[i] * g[m - i] where indices are valid
        # In 'full' mode, c_full[m] = sum_i h[i] * g[m-i+N-1] ... 
        # This is tricky. Let me use the direct formula.
        
        # Actually, let's just compute the gradient numerically or use a 
        # different approach. For now, let's compute gradients directly.
        
        # C_k for k > 0: C_k = sum_{i=0}^{N-1-k} h[i] * (1-h[i+k])
        # C_k for k < 0: C_k = sum_{i=-k}^{N-1} h[i] * (1-h[i+k])
        
        # Let's build the gradient matrix.
        num_shifts = len(c_all)
        grad = np.zeros((N, num_shifts))
        
        # c_all[0:N-1] corresponds to shifts k = -(N-1), ..., -1
        # c_all[N-1:2N-2] corresponds to shifts k = 1, ..., N-1
        
        # For shift k (positive): C_k = sum_{i=0}^{N-1-k} h[i] * (1-h[i+k])
        # dC_k/dh_j = (1-h[j+k]) if j+k <= N-1 and j >= 0
        #             -h[j-k] if j-k >= 0 and j < N
        #             (both if valid)
        
        for idx, k in enumerate(range(1, N)):
            col = N - 1 + idx  # index in c_all
            # C_k = sum_{i=0}^{N-1-k} h[i] * (1-h[i+k])
            for i in range(N - k):
                # d/dh[i] of h[i]*(1-h[i+k]) = (1-h[i+k])
                grad[i, col] += (1.0 - h[i+k])
                # d/dh[i+k] of h[i]*(1-h[i+k]) = -h[i]
                grad[i+k, col] -= h[i]
        
        for idx, k in enumerate(range(1, N)):
            col = idx  # index in c_all for negative shifts
            # C_{-k} = sum_{i=k}^{N-1} h[i] * (1-h[i-k])
            for i in range(k, N):
                # d/dh[i] of h[i]*(1-h[i-k]) = (1-h[i-k])
                grad[i, col] += (1.0 - h[i-k])
                # d/dh[i-k] of h[i]*(1-h[i-k]) = -h[i]
                grad[i-k, col] -= h[i]
        
        return c_all, grad

    def objective_lse_and_grad(h, beta, lambda_sum):
        """
        Objective: LSE(correlations) + penalty(sum - N/2)^2
        Returns (value, gradient)
        """
        c_all, grad = correlations_and_grad(h)
        
        # LogSumExp for numerical stability
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500
        exps = np.exp(beta * (c_all - max_c))
        sum_exps = np.sum(exps)
        lse = max_c + np.log(sum_exps) / beta
        
        # Gradient of LSE: dLSE/dh_j = (1/beta) * sum_k exps[k] * dC_k/dh_j / sum_exps
        weights = exps / sum_exps
        grad_lse = (1.0 / beta) * (grad.T @ weights)
        
        # Sum penalty
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - N/2)**2
        grad_sum = 2.0 * lambda_sum * (sum_h - N/2) * np.ones(N)
        
        value = lse + sum_pen
        gradient = grad_lse + grad_sum
        
        return value, gradient

    def objective_epigraph_and_grad(h, t, lambda_sum):
        """
        Epigraph formulation: minimize t + penalty(sum - N/2)^2
        with smooth max approximation.
        Returns (value, grad_h, grad_t)
        """
        c_all, grad = correlations_and_grad(h)
        
        # Smooth max using LSE
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500
        beta = 50.0
        exps = np.exp(beta * (c_all - max_c))
        sum_exps = np.sum(exps)
        smooth_max = max_c + np.log(sum_exps) / beta
        
        # Gradient of smooth max w.r.t. h
        weights = exps / sum_exps
        grad_h = (1.0 / beta) * (grad.T @ weights)
        
        # Sum penalty
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - N/2)**2
        grad_sum = 2.0 * lambda_sum * (sum_h - N/2) * np.ones(N)
        
        # Total objective: smooth_max + sum_pen (we minimize this, t is auxiliary)
        # Actually, let's use: max(0, C_k - t) formulation
        # Better: just use LSE as a proxy for max
        
        value = smooth_max + sum_pen
        grad_total = grad_h + grad_sum
        
        return value, grad_total

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = N/2.
        """
        h = np.array(h, dtype=float).copy()
        
        for _ in range(200):
            current_sum = np.sum(h)
            diff = N/2 - current_sum
            
            if abs(diff) < 1e-14:
                break
            
            if diff > 0:
                mask = h < 1.0 - 1e-15
                if not np.any(mask):
                    break
                n_incr = np.sum(mask)
                step = diff / n_incr
                h[mask] = np.minimum(h[mask] + step, 1.0)
            else:
                mask = h > 1e-15
                if not np.any(mask):
                    break
                n_decr = np.sum(mask)
                step = diff / n_decr
                h[mask] = np.maximum(h[mask] + step, 0.0)
        
        h = np.clip(h, 0.0, 1.0)
        
        current_sum = np.sum(h)
        if abs(current_sum - N/2) > 1e-12:
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
        """Optimize using LSE surrogate with analytic gradient."""
        bounds = [(0, 1) for _ in range(N)]
        
        def obj_and_grad(h):
            return objective_lse_and_grad(h, beta, lambda_sum)

        try:
            res = scipy.optimize.minimize(
                lambda h: obj_and_grad(h)[0], 
                h0, 
                jac=lambda h: obj_and_grad(h)[1],
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)

    rng = np.random.default_rng(42)
    
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
    
    # 10-13. Cosine-modulated initializations (from knowledge: low-freq cosine helps)
    for i in range(4):
        freq = (i + 1)  # periods across N
        phase = rng.uniform(0, 2*np.pi)
        h_cos = 0.5 + 0.3 * np.cos(2*np.pi*freq*np.arange(N)/N + phase)
        h_cos = np.clip(h_cos, 0, 1)
        h_cos = h_cos / np.sum(h_cos) * (N/2)
        h_cos = np.clip(h_cos, 0, 1)
        h_cos = project_to_feasible(h_cos)
        
        beta = 10.0 + i * 5.0
        lambda_sum = 150.0 + i * 50.0
        
        restarts.append((h_cos, 'lse', (beta, lambda_sum), 300))
    
    for idx, (h0, method, params, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit:
            break
            
        if method == 'lse':
            beta, lambda_sum = params
            h_opt = optimize_lse(h0, beta, lambda_sum, max_iter)
        
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
        h_polish = best_h.copy()
        
        for beta_polish in [30.0, 50.0, 100.0, 200.0]:
            if time.time() - start_time > time_limit:
                break
            
            h_opt = optimize_lse(h_polish, beta_polish, 500.0, 1000, ftol=1e-11, gtol=1e-11)
            
            current_score = score_h(h_opt)
            
            if np.isclose(np.sum(h_opt), N/2, atol=1e-10) and \
               np.all(h_opt >= -1e-12) and np.all(h_opt <= 1 + 1e-12):
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
                    h_polish = h_opt.copy()
    
    # Ensure final feasibility
    best_h = project_to_feasible(best_h)
    
    return best_h
