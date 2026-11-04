import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    N = 100
    target_sum = N / 2.0
    
    def get_correlations(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # The problem usually considers the central part or all shifts.
        # The current code uses c_nonzero = np.concatenate([c[:N-1], c[N:]])
        # This excludes the lag 0? 
        # np.correlate(h, g, 'full') has length 2N-1.
        # Index N-1 is lag 0.
        # The current code excludes index N-1 (lag 0).
        # Why? 
        # "minimize the maximum overlap of h with shifted 1-h"
        # Usually "shifted" implies non-zero shift.
        # If lag 0 is included, C_0 = sum h_i(1-h_i).
        # If h is binary, C_0 = 0.
        # If h is 0.5, C_0 = N * 0.25 = 25.
        # If we exclude lag 0, the baseline for h=0.5 is 25 (for other lags).
        # Let's stick to the current definition: exclude lag 0.
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def score_h(h):
        c_all = get_correlations(h)
        return np.max(c_all)

    def lse_smooth_max(c, beta):
        max_c = np.max(c)
        if max_c > 500:
            max_c = 500
        return max_c + np.log(np.sum(np.exp(beta * (c - max_c)))) / beta

    def objective(h, beta, lambda_bin):
        c_all = get_correlations(h)
        lse_val = lse_smooth_max(c_all, beta)
        # Binarization penalty: encourages h_i to be 0 or 1
        # sum h_i(1-h_i) is 0 if binary, max if 0.5
        bin_pen = lambda_bin * np.sum(h * (1 - h))
        return lse_val + bin_pen

    def optimize_instance(h0, beta, lambda_bin, max_iter, method='SLSQP'):
        bounds = [(0, 1) for _ in range(N)]
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - target_sum, 
             'jac': lambda h: np.ones(N)}
        ]
        
        def obj(h):
            return objective(h, beta, lambda_bin)
            
        def grad_obj(h):
            # Numerical gradient for simplicity, or analytic
            # Analytic grad of LSE is (sum exp(beta(c-max)) * dc/dh) / sum exp(beta(c-max))
            # This is complex. Let's use numerical or let SLSQP compute it.
            # SLSQP can compute finite differences.
            return None

        try:
            res = scipy.optimize.minimize(
                obj,
                h0,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        h_opt = np.clip(h_opt, 0, 1)
        
        # Correct sum
        for _ in range(50):
            diff = target_sum - np.sum(h_opt)
            if abs(diff) < 1e-14:
                break
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            
        return h_opt

    def try_binary_rounding(h):
        # Round to nearest integer, then adjust to have exactly N/2 ones
        h_rounded = np.round(h)
        # Ensure sum is N/2
        num_ones = np.sum(h_rounded == 1)
        target_ones = int(target_sum)
        
        if num_ones < target_ones:
            # Need more ones. Pick indices with highest h value among 0s
            zeros = np.where(h_rounded == 0)[0]
            if len(zeros) > 0:
                sort_idx = np.argsort(-h[zeros])
                for i in sort_idx[:target_ones - num_ones]:
                    h_rounded[zeros[i]] = 1
        elif num_ones > target_ones:
            # Need fewer ones. Pick indices with lowest h value among 1s
            ones = np.where(h_rounded == 1)[0]
            if len(ones) > 0:
                sort_idx = np.argsort(h[ones])
                for i in sort_idx[:num_ones - target_ones]:
                    h_rounded[ones[i]] = 0
                    
        return h_rounded.astype(float)

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)
    
    rng = np.random.default_rng(42)
    
    # Schedule of (beta, lambda_bin, max_iter)
    # Start with low lambda to find structure, then increase to binarize
    schedule = [
        (10.0, 0.0, 200),
        (20.0, 5.0, 200),
        (30.0, 20.0, 200),
        (50.0, 100.0, 300),
        (100.0, 500.0, 500),
        (200.0, 2000.0, 1000)
    ]
    
    # Initializations
    inits = [
        np.full(N, 0.5),
        rng.uniform(0.2, 0.8, N),
        np.where(rng.uniform(0, 1, N) > 0.5, 0.8, 0.2),
        np.where(rng.uniform(0, 1, N) > 0.5, 1.0, 0.0) # Binary init
    ]
    
    for h_init in inits:
        if time.time() - start_time > time_limit - 10:
            break
            
        h_curr = h_init.copy()
        h_curr = h_curr / np.sum(h_curr) * target_sum
        h_curr = np.clip(h_curr, 0, 1)
        
        for beta, lambda_bin, max_iter in schedule:
            if time.time() - start_time > time_limit - 5:
                break
                
            h_opt = optimize_instance(h_curr, beta, lambda_bin, max_iter)
            
            # Check continuous score
            score_cont = score_h(h_opt)
            if score_cont < best_score:
                best_score = score_cont
                best_h = h_opt.copy()
                
            # Try binary rounding
            h_bin = try_binary_rounding(h_opt)
            score_bin = score_h(h_bin)
            if score_bin < best_score:
                best_score = score_bin
                best_h = h_bin.copy()
                
            h_curr = h_opt.copy()
            
    # Final local search on the best found (continuous or binary)
    if time.time() - start_time < time_limit - 5:
        # If best is binary, do swap search
        if np.allclose(best_h, np.round(best_h)):
            h_search = best_h.copy()
            current_score = score_h(h_search)
            for _ in range(2000):
                if time.time() - start_time > time_limit - 2:
                    break
                i, j = np.random.choice(N, 2, replace=False)
                if h_search[i] == h_search[j]:
                    continue
                h_new = h_search.copy()
                h_new[i], h_new[j] = h_new[j], h_new[i]
                new_score = score_h(h_new)
                if new_score < current_score - 1e-10:
                    h_search = h_new
                    current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h_search.copy()
        else:
            # Continuous local search
            h_search = best_h.copy()
            current_score = score_h(h_search)
            for _ in range(1000):
                if time.time() - start_time > time_limit - 2:
                    break
                # Random small perturbation
                idx = np.random.randint(N)
                delta = np.random.uniform(-0.1, 0.1)
                h_new = h_search.copy()
                h_new[idx] += delta
                # Project to sum and bounds
                h_new = np.clip(h_new, 0, 1)
                diff = target_sum - np.sum(h_new)
                h_new += diff / N
                h_new = np.clip(h_new, 0, 1)
                
                new_score = score_h(h_new)
                if new_score < current_score - 1e-10:
                    h_search = h_new
                    current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h_search.copy()

    # Ensure final constraints
    best_h = np.clip(best_h, 0, 1)
    for _ in range(50):
        diff = target_sum - np.sum(best_h)
        if abs(diff) < 1e-14:
            break
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
        
    return best_h
