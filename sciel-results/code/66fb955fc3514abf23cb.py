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
        # Exclude lag 0 (index N-1)
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
        # Binarization penalty
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
        h_rounded = np.round(h)
        num_ones = np.sum(h_rounded == 1)
        target_ones = int(target_sum)
        
        if num_ones < target_ones:
            zeros = np.where(h_rounded == 0)[0]
            if len(zeros) > 0:
                sort_idx = np.argsort(-h[zeros])
                for i in sort_idx[:target_ones - num_ones]:
                    h_rounded[zeros[i]] = 1
        elif num_ones > target_ones:
            ones = np.where(h_rounded == 1)[0]
            if len(ones) > 0:
                sort_idx = np.argsort(h[ones])
                for i in sort_idx[:num_ones - target_ones]:
                    h_rounded[ones[i]] = 0
                    
        return h_rounded.astype(float)

    def local_search_binary(h_bin, max_iter=10000):
        """
        Local search on binary sequences.
        Moves: Swap a 1 and a 0.
        """
        h_curr = h_bin.copy()
        curr_score = score_h(h_curr)
        
        # Precompute indices of 1s and 0s for fast sampling?
        # Or just random sampling.
        
        no_improve_count = 0
        for i in range(max_iter):
            if time.time() - start_time > time_limit - 2:
                break
                
            # Pick random 1 and random 0
            ones = np.where(h_curr == 1)[0]
            zeros = np.where(h_curr == 0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
                
            i1 = np.random.choice(ones)
            i0 = np.random.choice(zeros)
            
            # Swap
            h_new = h_curr.copy()
            h_new[i1] = 0
            h_new[i0] = 1
            
            new_score = score_h(h_new)
            
            if new_score < curr_score - 1e-10:
                h_curr = h_new
                curr_score = new_score
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Simple restart logic if stuck
                if no_improve_count > 100:
                    # Random perturbation: swap 5 random pairs
                    for _ in range(5):
                        ones = np.where(h_curr == 1)[0]
                        zeros = np.where(h_curr == 0)[0]
                        if len(ones) == 0 or len(zeros) == 0:
                            break
                        i1 = np.random.choice(ones)
                        i0 = np.random.choice(zeros)
                        h_curr[i1] = 0
                        h_curr[i0] = 1
                    curr_score = score_h(h_curr)
                    no_improve_count = 0

        return h_curr, curr_score

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)
    
    rng = np.random.default_rng(42)
    
    # Schedule for continuous optimization
    schedule = [
        (10.0, 0.0, 100),
        (20.0, 5.0, 100),
        (50.0, 50.0, 200),
        (100.0, 200.0, 300),
        (200.0, 1000.0, 500)
    ]
    
    # Initializations
    inits = [
        np.full(N, 0.5),
        rng.uniform(0.2, 0.8, N),
        np.where(rng.uniform(0, 1, N) > 0.5, 0.8, 0.2),
        np.where(rng.uniform(0, 1, N) > 0.5, 1.0, 0.0)
    ]
    
    # Add some structured initializations
    # Legendre-like: x_i = 1 if i^2 is quadratic residue mod p?
    # Simple pattern: 101010...
    inits.append(np.tile([1.0, 0.0], N//2))
    inits.append(np.tile([1.0, 1.0, 0.0, 0.0], N//4))
    
    for h_init in inits:
        if time.time() - start_time > time_limit - 10:
            break
            
        h_curr = h_init.copy()
        # Normalize sum
        if np.sum(h_curr) > 0:
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
            
            # If we have a binary solution, run local search on it
            if np.allclose(h_bin, np.round(h_bin)):
                if time.time() - start_time < time_limit - 5:
                    h_ls, score_ls = local_search_binary(h_bin, max_iter=5000)
                    if score_ls < best_score:
                        best_score = score_ls
                        best_h = h_ls.copy()

    # Final global binary local search from best found
    if time.time() - start_time < time_limit - 5:
        # Ensure best_h is binary for local search
        h_bin_best = try_binary_rounding(best_h)
        if score_h(h_bin_best) < best_score:
            best_h = h_bin_best
            best_score = score_h(best_h)
            
        if np.allclose(best_h, np.round(best_h)):
            h_ls, score_ls = local_search_binary(best_h, max_iter=20000)
            if score_ls < best_score:
                best_score = score_ls
                best_h = h_ls.copy()
        else:
            # Continuous polish
            h_opt = optimize_instance(best_h, 200.0, 1000.0, 500)
            score_opt = score_h(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
            
            h_bin = try_binary_rounding(h_opt)
            score_bin = score_h(h_bin)
            if score_bin < best_score:
                best_score = score_bin
                best_h = h_bin.copy()

    # Ensure final constraints
    best_h = np.clip(best_h, 0, 1)
    for _ in range(50):
        diff = target_sum - np.sum(best_h)
        if abs(diff) < 1e-14:
            break
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
        
    return best_h
