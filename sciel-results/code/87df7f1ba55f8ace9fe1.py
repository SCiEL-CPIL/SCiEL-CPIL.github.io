import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s limit
    
    N = 100
    target_sum = N / 2.0
    
    def get_correlations_linear(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def score_h(h):
        c_all = get_correlations_linear(h)
        return np.max(c_all)
    
    def lse_smooth_max(c, beta):
        max_c = np.max(c)
        if max_c > 500:
            max_c = 500
        return max_c + np.log(np.sum(np.exp(beta * (c - max_c)))) / beta
    
    def objective(h, beta, lambda_sum):
        c_all = get_correlations_linear(h)
        lse_val = lse_smooth_max(c_all, beta)
        sum_diff = np.sum(h) - target_sum
        pen = lambda_sum * sum_diff**2
        return lse_val + pen
    
    def optimize_instance(h0, beta, lambda_sum, max_iter, method='L-BFGS-B'):
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            return objective(h, beta, lambda_sum)
        
        try:
            res = scipy.optimize.minimize(
                obj,
                h0,
                method=method,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        h_opt = np.clip(h_opt, 0, 1)
        
        # Correct sum to be exactly N/2
        for _ in range(50):
            current_sum = np.sum(h_opt)
            diff = target_sum - current_sum
            if abs(diff) < 1e-14:
                break
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
        
        return h_opt

    def local_search(h, max_swaps=1000):
        """
        Perform binary swap local search to improve the score.
        """
        h = h.copy()
        current_score = score_h(h)
        improvements = 0
        for _ in range(max_swaps):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Randomly select two indices to swap
            i, j = np.random.choice(N, 2, replace=False)
            
            # Try swapping values
            h_new = h.copy()
            h_new[i], h_new[j] = h_new[j], h_new[i]
            
            # Check if sum is still valid (it is, since we swap)
            # Check bounds (they are, since we swap existing values)
            
            new_score = score_h(h_new)
            
            if new_score < current_score - 1e-10:
                h = h_new
                current_score = new_score
                improvements += 1
            else:
                # If no improvement in a while, we could break, but let's just continue
                pass
                
        return h

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)
    
    rng = np.random.default_rng(42)
    
    # Multi-restart strategy with diverse initializations
    restart_configs = []
    
    # 1. Constant start (baseline)
    restart_configs.append((np.full(N, 0.5), 10.0, 500.0, 300, 'L-BFGS-B'))
    
    # 2-5. Smooth random perturbations
    for i in range(4):
        h_rand = rng.uniform(0.2, 0.8, size=N)
        h_rand = h_rand / np.sum(h_rand) * target_sum
        h_rand = np.clip(h_rand, 0, 1)
        beta = 10.0 + i * 10.0
        lambda_sum = 500.0 + i * 200.0
        restart_configs.append((h_rand, beta, lambda_sum, 300, 'L-BFGS-B'))
    
    # 6-9. Bimodal initializations
    for i in range(4):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.8, 0.2)
        h_rand = h_rand / np.sum(h_rand) * target_sum
        h_rand = np.clip(h_rand, 0, 1)
        beta = 20.0 + i * 10.0
        lambda_sum = 1000.0 + i * 200.0
        restart_configs.append((h_rand, beta, lambda_sum, 300, 'L-BFGS-B'))
    
    # 10-13. High-beta refinement from previous best
    for i in range(4):
        h0 = best_h + rng.normal(0, 0.05, N)
        h0 = np.clip(h0, 0, 1)
        h0 = h0 / np.sum(h0) * target_sum
        h0 = np.clip(h0, 0, 1)
        beta = 30.0 + i * 15.0
        lambda_sum = 2000.0 + i * 500.0
        restart_configs.append((h0, beta, lambda_sum, 200, 'L-BFGS-B'))
    
    for idx, (h0, beta, lambda_sum, max_iter, method) in enumerate(restart_configs):
        if time.time() - start_time > time_limit - 5:
            break
        
        h_opt = optimize_instance(h0, beta, lambda_sum, max_iter, method)
        
        # Verify constraints
        if not np.isclose(np.sum(h_opt), target_sum, atol=1e-8):
            continue
        if not (np.all(h_opt >= -1e-10) and np.all(h_opt <= 1 + 1e-10)):
            continue
        
        # Apply local search
        h_opt = local_search(h_opt, max_swaps=500)
        
        # Re-verify constraints after local search (swaps preserve sum and bounds)
        if not np.isclose(np.sum(h_opt), target_sum, atol=1e-8):
            continue
        if not (np.all(h_opt >= -1e-10) and np.all(h_opt <= 1 + 1e-10)):
            continue
            
        current_score = score_h(h_opt)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
    
    # Final high-precision polish
    if time.time() - start_time < time_limit - 10:
        h0 = best_h.copy()
        beta_final = 50.0
        lambda_final = 5000.0
        # Use SLSQP for better local refinement
        h_polished = optimize_instance(h0, beta_final, lambda_final, 1000, 'SLSQP')
        
        if np.isclose(np.sum(h_polished), target_sum, atol=1e-10) and \
           np.all(h_polished >= -1e-10) and np.all(h_polished <= 1 + 1e-10):
            
            # Apply local search again
            h_polished = local_search(h_polished, max_swaps=2000)
            
            if np.isclose(np.sum(h_polished), target_sum, atol=1e-10) and \
               np.all(h_polished >= -1e-10) and np.all(h_polished <= 1 + 1e-10):
                polished_score = score_h(h_polished)
                if polished_score < best_score:
                    best_score = polished_score
                    best_h = h_polished.copy()
    
    # Ensure final solution satisfies constraints
    best_h = np.clip(best_h, 0, 1)
    for _ in range(50):
        diff = target_sum - np.sum(best_h)
        if abs(diff) < 1e-14:
            break
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
