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
    
    def project_to_feasible(h):
        """Project h onto the feasible set: sum = target_sum, 0 <= h <= 1"""
        h = h.copy()
        for _ in range(100):
            # First ensure bounds
            h = np.clip(h, 0, 1)
            # Then adjust sum
            current_sum = np.sum(h)
            diff = target_sum - current_sum
            if abs(diff) < 1e-15:
                break
            # Distribute adjustment only to elements not at bounds
            can_increase = h < 1.0 - 1e-15
            can_decrease = h > 1e-15
            
            if diff > 0:
                # Need to increase sum
                if not np.any(can_increase):
                    break
                n_inc = np.sum(can_increase)
                step = diff / n_inc
                h[can_increase] = np.clip(h[can_increase] + step, 0, 1)
            else:
                # Need to decrease sum
                if not np.any(can_decrease):
                    break
                n_dec = np.sum(can_decrease)
                step = diff / n_dec
                h[can_decrease] = np.clip(h[can_decrease] + step, 0, 1)
        
        # Final exact correction
        diff = target_sum - np.sum(h)
        if abs(diff) > 1e-14:
            # Find elements with room to adjust
            can_adjust = (h > 1e-15) & (h < 1.0 - 1e-15)
            if np.any(can_adjust):
                n_adj = np.sum(can_adjust)
                h[can_adjust] += diff / n_adj
                h = np.clip(h, 0, 1)
                # One more pass for exactness
                diff2 = target_sum - np.sum(h)
                if abs(diff2) > 1e-16:
                    h += diff2 / N
                    h = np.clip(h, 0, 1)
        
        return h
    
    def epigraph_objective(x, beta):
        """x = [h (N values), t (1 value)]"""
        h = x[:N]
        t = x[N]
        c_all = get_correlations_linear(h)
        lse_val = lse_smooth_max(c_all, beta)
        # Epigraph: t >= max(c), penalize if t < lse_val
        epigraph_pen = np.maximum(0, lse_val - t) ** 2
        # Sum constraint
        sum_diff = np.sum(h) - target_sum
        sum_pen = 1000.0 * sum_diff ** 2
        return t + epigraph_pen + sum_pen
    
    def optimize_with_epigraph(h0, beta, max_iter):
        """Optimize using epigraph formulation"""
        t0 = score_h(h0) + 0.1
        x0 = np.concatenate([h0, [t0]])
        
        bounds = [(0, 1)] * N + [(0, 2.0)]
        
        try:
            res = scipy.optimize.minimize(
                lambda x: epigraph_objective(x, beta),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 100}
            )
            x_opt = res.x
            h_opt = x_opt[:N]
        except Exception:
            h_opt = h0.copy()
        
        h_opt = project_to_feasible(h_opt)
        return h_opt
    
    def optimize_instance(h0, beta, lambda_sum, max_iter):
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            c_all = get_correlations_linear(h)
            lse_val = lse_smooth_max(c_all, beta)
            sum_diff = np.sum(h) - target_sum
            pen = lambda_sum * sum_diff**2
            return lse_val + pen
        
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
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    # Phase 1: Combinatorial Search (Simulated Annealing) on Binary Sequences
    # This finds a good structural "skeleton" with low max overlap.
    
    rng = np.random.default_rng(42)
    
    # Initialize with a random binary sequence with exactly N/2 ones
    h_bin = np.zeros(N)
    indices = rng.permutation(N)
    h_bin[indices[:N//2]] = 1.0
    
    best_h_bin = h_bin.copy()
    best_score_bin = score_h(h_bin)
    
    # Multiple restarts with different seeds for diversity
    best_overall_h = best_h_bin.copy()
    best_overall_score = best_score_bin
    
    for restart in range(5):
        if time.time() - start_time > time_limit - 40:
            break
        
        if restart > 0:
            h_bin = np.zeros(N)
            indices = rng.permutation(N)
            h_bin[indices[:N//2]] = 1.0
            current_h_bin = h_bin.copy()
            current_score_bin = score_h(h_bin)
        else:
            current_h_bin = h_bin.copy()
            current_score_bin = best_score_bin
        
        # Simulated Annealing with better cooling
        T = 5.0
        cooling_rate = 0.997
        min_T = 0.05
        stall_counter = 0
        max_stall = 500
        
        while T > min_T and time.time() - start_time < time_limit - 40 and stall_counter < max_stall:
            # Pick a 1 and a 0 to swap
            ones = np.where(current_h_bin == 1.0)[0]
            zeros = np.where(current_h_bin == 0.0)[0]
            
            if len(ones) == 0 or len(zeros) == 0:
                break
                
            i = ones[rng.integers(len(ones))]
            j = zeros[rng.integers(len(zeros))]
            
            # Swap
            current_h_bin[i], current_h_bin[j] = current_h_bin[j], current_h_bin[i]
            
            new_score = score_h(current_h_bin)
            delta = new_score - current_score_bin
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_score_bin = new_score
                stall_counter = 0
                if new_score < best_overall_score:
                    best_overall_score = new_score
                    best_overall_h = current_h_bin.copy()
            else:
                # Revert
                current_h_bin[i], current_h_bin[j] = current_h_bin[j], current_h_bin[i]
                stall_counter += 1
            
            T *= cooling_rate
        
        if current_score_bin < best_score_bin:
            best_h_bin = current_h_bin.copy()
            best_score_bin = current_score_bin
    
    best_h_bin = best_overall_h
    best_score_bin = best_overall_score
        
    # Phase 2: Continuous Refinement
    # Use the best binary sequence as a starting point for L-BFGS-B
    # This allows the values to move away from 0/1 to potentially lower the max peak further.
    
    h0 = best_h_bin.copy()
    
    # Run a few refinements with different beta
    best_h = h0
    best_score = best_score_bin
    
    for beta in [5.0, 10.0, 20.0, 50.0]:
        if time.time() - start_time > time_limit - 20:
            break
            
        h_opt = optimize_instance(h0, beta, 1000.0, 300)
        
        if np.isclose(np.sum(h_opt), target_sum, atol=1e-8) and \
           np.all(h_opt >= -1e-10) and np.all(h_opt <= 1 + 1e-10):
            
            current_score = score_h(h_opt)
            if current_score < best_score:
                best_score = current_score
                best_h = h_opt.copy()
                h0 = best_h.copy() # Use improved solution for next refinement
    
    # Phase 3: Epigraph-based high-precision polish
    if time.time() - start_time < time_limit - 10:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() - start_time > time_limit - 5:
                break
                
            h_opt = optimize_with_epigraph(best_h, beta, 500)
            
            if np.isclose(np.sum(h_opt), target_sum, atol=1e-8) and \
               np.all(h_opt >= -1e-10) and np.all(h_opt <= 1 + 1e-10):
                
                current_score = score_h(h_opt)
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
    
    # Phase 4: Final ultra-high-precision polish with very tight tolerances
    if time.time() - start_time < time_limit - 3:
        for beta in [1000.0]:
            if time.time() - start_time > time_limit - 1:
                break
                
            h_opt = optimize_instance(best_h, beta, 50000.0, 1000)
            
            if np.isclose(np.sum(h_opt), target_sum, atol=1e-8) and \
               np.all(h_opt >= -1e-10) and np.all(h_opt <= 1 + 1e-10):
                
                current_score = score_h(h_opt)
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
    
    # Final constraint check and adjustment
    best_h = project_to_feasible(best_h)
        
    return best_h
