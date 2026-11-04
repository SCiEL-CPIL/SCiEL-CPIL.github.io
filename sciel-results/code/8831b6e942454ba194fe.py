import numpy as np
import scipy.optimize
import time
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s limit
    
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
        if max_c > 100:
            return max_c
        shifted = beta * (c - max_c)
        shifted = np.clip(shifted, -500, 0)
        log_sum = np.log(np.sum(np.exp(shifted)))
        return max_c + log_sum / beta

    def epigraph_objective(h, t, beta, lambda_sum):
        c_all = get_correlations_linear(h)
        violations = np.maximum(c_all - t, 0)
        soft_viol = np.sum(np.logaddexp(0, beta * (c_all - t))) / beta
        sum_diff = np.sum(h) - target_sum
        pen = lambda_sum * sum_diff**2
        return t + soft_viol + pen

    def objective(h, beta, lambda_sum):
        c_all = get_correlations_linear(h)
        lse_val = lse_smooth_max(c_all, beta)
        sum_diff = np.sum(h) - target_sum
        pen = lambda_sum * sum_diff**2
        return lse_val + pen

    def project_exact(y, target_sum):
        """
        Project y onto {h : 0 <= h_i <= 1, sum(h) = target_sum}
        using bisection on lambda: h_i = clip(y_i - lambda, 0, 1)
        """
        # If sum is already within bounds and matches, return
        if np.all(y >= 0) and np.all(y <= 1) and np.isclose(np.sum(y), target_sum, atol=1e-12):
            return y.copy()
        
        # Bisection range for lambda
        # h_i = clip(y_i - lambda, 0, 1)
        # sum h_i = target_sum
        # If lambda is very small (negative), h_i -> 1 for large y_i
        # If lambda is very large, h_i -> 0
        
        # Initial bounds
        lo = np.min(y) - 2.0
        hi = np.max(y) + 2.0
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            h_proj = np.clip(y - mid, 0.0, 1.0)
            s = np.sum(h_proj)
            if s > target_sum:
                # Need to reduce sum, increase lambda
                lo = mid
            else:
                # Need to increase sum, decrease lambda
                hi = mid
                
        h_final = np.clip(y - (lo + hi) / 2.0, 0.0, 1.0)
        # Final adjustment for floating point precision
        diff = target_sum - np.sum(h_final)
        if abs(diff) > 1e-12:
            # Distribute remaining diff to elements not at bounds
            mask = (h_final > 0) & (h_final < 1)
            if np.sum(mask) > 0:
                h_final[mask] += diff / np.sum(mask)
                h_final = np.clip(h_final, 0.0, 1.0)
            else:
                # If all at bounds, we are infeasible or degenerate, just clip
                h_final = np.clip(h_final, 0.0, 1.0)
        return h_final

    def optimize_instance(h0, beta, lambda_sum, max_iter):
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            return objective(h, beta, lambda_sum)
        
        try:
            res = scipy.optimize.minimize(
                obj,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        h_opt = np.clip(h_opt, 0, 1)
        h_opt = project_exact(h_opt, target_sum)
        return h_opt
    
    def optimize_epigraph(h0, beta, lambda_sum, max_iter):
        c_init = get_correlations_linear(h0)
        t0 = np.max(c_init) + 0.1
        
        x0 = np.concatenate([h0, [t0]])
        bounds_e = [(0, 1)] * N + [(-1, 2)]
        
        def obj_e(x):
            h = x[:N]
            t = x[N]
            return epigraph_objective(h, t, beta, lambda_sum)
        
        try:
            res = scipy.optimize.minimize(
                obj_e,
                x0,
                method='L-BFGS-B',
                bounds=bounds_e,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = res.x[:N]
        except Exception:
            h_opt = h0.copy()
        
        h_opt = np.clip(h_opt, 0, 1)
        h_opt = project_exact(h_opt, target_sum)
        return h_opt

    def coordinate_descent_polish(h, max_sweeps=10):
        """
        Polish using coordinate descent with exact max-correlation objective.
        """
        current_h = h.copy()
        current_score = score_h(current_h)
        
        for sweep in range(max_sweeps):
            if time.time() - start_time > time_limit - 5:
                break
                
            improved = False
            for i in range(N):
                if time.time() - start_time > time_limit - 5:
                    break
                    
                # Fix all other coordinates
                h_fixed = current_h.copy()
                
                def obj_1d(h_i_val):
                    h_candidate = h_fixed.copy()
                    h_candidate[i] = h_i_val
                    # Check bounds (minimize_scalar handles bounds, but we clip for safety)
                    if h_i_val < 0 or h_i_val > 1:
                        return 1e10
                    return score_h(h_candidate)
                
                # Minimize obj_1d in [0, 1]
                try:
                    res = scipy.optimize.minimize_scalar(
                        obj_1d,
                        bounds=(0, 1),
                        method='bounded',
                        options={'xatol': 1e-8}
                    )
                    new_h_i = res.x
                except Exception:
                    new_h_i = current_h[i]
                
                # Update
                if abs(new_h_i - current_h[i]) > 1e-10:
                    current_h[i] = new_h_i
                    # Project to maintain sum constraint? 
                    # Coordinate descent changes sum. We must project.
                    current_h = project_exact(current_h, target_sum)
                    
                    new_score = score_h(current_h)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                    else:
                        # If it didn't improve or got worse due to projection side-effects, 
                        # we might want to revert? But projection is necessary for feasibility.
                        # Usually, if the 1D min was good, the projection should keep it close.
                        pass
            
            if not improved:
                break
                
        return current_h, current_score

    # Phase 1: Combinatorial Search (Simulated Annealing) on Binary Sequences
    rng = np.random.default_rng(42)
    
    h_bin = np.zeros(N)
    indices = rng.permutation(N)
    h_bin[indices[:N//2]] = 1.0
    
    best_h_bin = h_bin.copy()
    best_score_bin = score_h(h_bin)
    
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start_time < time_limit - 40:
        ones = np.where(h_bin == 1.0)[0]
        zeros = np.where(h_bin == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        
        h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
        
        new_score = score_h(h_bin)
        delta = new_score - best_score_bin
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            best_h_bin = h_bin.copy()
            best_score_bin = new_score
        else:
            h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
            
        T *= cooling_rate
        
    # Phase 2: Continuous Refinement
    h0 = best_h_bin.copy()
    best_h = h0
    best_score = best_score_bin
    
    for beta in [10.0, 30.0, 100.0]:
        if time.time() - start_time > time_limit - 15:
            break
        
        h_opt1 = optimize_instance(h0, beta, 5000.0, 300)
        current_score1 = score_h(h_opt1)
        if current_score1 < best_score:
            best_score = current_score1
            best_h = h_opt1.copy()
            h0 = best_h.copy()
        
        if time.time() - start_time < time_limit - 15:
            h_opt2 = optimize_epigraph(h0, beta, 5000.0, 300)
            current_score2 = score_h(h_opt2)
            if current_score2 < best_score:
                best_score = current_score2
                best_h = h_opt2.copy()
                h0 = best_h.copy()
    
    # Phase 3: High-precision polish pass (L-BFGS with high beta)
    if time.time() - start_time < time_limit - 10:
        for beta in [300.0, 1000.0]:
            if time.time() - start_time > time_limit - 5:
                break
            
            h_opt1 = optimize_instance(best_h, beta, 50000.0, 800)
            current_score1 = score_h(h_opt1)
            if current_score1 < best_score:
                best_score = current_score1
                best_h = h_opt1.copy()
            
            if time.time() - start_time < time_limit - 5:
                h_opt2 = optimize_epigraph(best_h, beta, 50000.0, 800)
                current_score2 = score_h(h_opt2)
                if current_score2 < best_score:
                    best_score = current_score2
                    best_h = h_opt2.copy()
    
    # Phase 4: Coordinate Descent Polish
    if time.time() - start_time < time_limit - 2:
        best_h, best_score = coordinate_descent_polish(best_h, max_sweeps=5)

    # Final constraint check and exact projection
    best_h = project_exact(best_h, target_sum)
    
    return best_h
