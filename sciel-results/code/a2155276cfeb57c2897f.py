import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    N = 100
    target_sum = N / 2.0
    
    def get_correlations_linear(h):
        g = 1.0 - h
        c_full = np.correlate(g, h, mode='full')
        c_nonzero = np.concatenate([c_full[:N-1], c_full[N:]])
        return c_nonzero

    def max_overlap(h):
        c = get_correlations_linear(h)
        return np.max(c)
    
    def smooth_max(c, temp=0.05):
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:
            return m * temp
        return temp * (m + np.log(np.sum(np.exp(scaled - m))))
    
    def project(h):
        h = np.clip(h, 0, 1)
        for _ in range(20):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0, 1)
        return h
    
    # Objective with epigraph formulation: minimize t + penalty
    # subject to c_i <= t for all i, 0<=h<=1, sum(h)=N/2
    # We use a smooth approximation of the constraint c_i <= t via log-sum-exp or soft max
    # Actually, the smooth max is already a good surrogate for t.
    # Let's refine the L-BFGS-B step with tighter tolerance and more iterations.
    
    def objective_smooth(h, temp=0.05):
        c = get_correlations_linear(h)
        val = smooth_max(c, temp)
        sum_err = np.sum(h) - target_sum
        return val + 10.0 * sum_err**2

    # Initialize
    rng = np.random.default_rng(42)
    h_curr = np.full(N, 0.5)
    best_h = h_curr.copy()
    best_score = max_overlap(best_h)
    
    # Simulated annealing
    T_init = 0.1
    T_final = 0.001
    n_iterations = 5000
    
    def find_clusters(h, threshold=0.05):
        clusters = []
        i = 0
        while i < N:
            j = i
            base_val = h[i]
            while j < N and abs(h[j] - base_val) < threshold:
                j += 1
            if j > i:
                clusters.append((i, j))
            i = j
        return clusters
    
    def cluster_move(h, clusters, rng):
        if not clusters:
            return h.copy()
        idx = rng.integers(0, len(clusters))
        i, j = clusters[idx]
        h_new = h.copy()
        shift = rng.uniform(-0.2, 0.2)
        h_new[i:j] += shift
        h_new = project(h_new)
        return h_new
    
    def cluster_flip(h, clusters, rng):
        if not clusters:
            return h.copy()
        idx = rng.integers(0, len(clusters))
        i, j = clusters[idx]
        h_new = h.copy()
        h_new[i:j] = 1.0 - h[i:j]
        h_new = project(h_new)
        return h_new

    for it in range(n_iterations):
        if time.time() - start_time > time_limit - 20:
            break
        
        progress = it / n_iterations
        T = T_init * (T_final / T_init) ** progress
        
        clusters = find_clusters(h_curr, threshold=0.05)
        
        move_type = rng.random()
        if move_type < 0.4:
            h_prop = cluster_move(h_curr, clusters, rng)
        elif move_type < 0.7:
            h_prop = cluster_flip(h_curr, clusters, rng)
        else:
            h_prop = h_curr.copy()
            i = rng.integers(0, N)
            h_prop[i] += rng.uniform(-0.1, 0.1)
            h_prop = project(h_prop)
        
        score_curr = max_overlap(h_curr)
        score_prop = max_overlap(h_prop)
        
        delta = score_prop - score_curr
        if delta < 0 or rng.random() < np.exp(-delta / T):
            h_curr = h_prop
        
        curr_score = max_overlap(h_curr)
        if curr_score < best_score:
            best_score = curr_score
            best_h = h_curr.copy()
    
    # High-precision polish with multiple temperatures and tighter tolerances
    if time.time() - start_time < time_limit - 5:
        h_init = best_h.copy()
        bounds = [(0, 1)] * N
        
        # Use a sequence of decreasing temperatures for smoother convergence
        for temp in [0.1, 0.05, 0.02, 0.01, 0.005]:
            if time.time() - start_time > time_limit - 2:
                break
            
            def obj_scipy(x, t=temp):
                return objective_smooth(x, t)
            
            try:
                res = minimize(obj_scipy, h_init, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 5000, 'maxfun': 50000, 'gtol': 1e-10, 'ftol': 1e-12})
                h_opt = res.x
                h_opt = project(h_opt)
                score_opt = max_overlap(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    h_init = h_opt.copy()
            except Exception:
                pass

    # Final coordinate descent polish
    if time.time() - start_time < time_limit - 1:
        h_polish = best_h.copy()
        for _ in range(1000):
            if time.time() - start_time > time_limit - 1:
                break
            improved = False
            for i in range(N):
                for step in [-0.01, 0.01]:
                    h_try = h_polish.copy()
                    h_try[i] += step
                    h_try = project(h_try)
                    if max_overlap(h_try) < max_overlap(h_polish) - 1e-12:
                        h_polish = h_try
                        improved = True
                        break
            if not improved:
                break
        if max_overlap(h_polish) < best_score:
            best_h = h_polish
            best_score = max_overlap(best_h)

    best_h = project(best_h)
    return best_h
