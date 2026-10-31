"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def compute_score(heights):
        """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.array(heights)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        # Autoconvolution (linear, full mode)
        conv = np.convolve(h, h, mode='full')
        linf = np.max(conv)
        if linf < 1e-12:
            return 0.0
        return l2_sq / (l1 * linf)
    
    def find_sidon_set(n, target_size):
        """Find a Sidon set (B2 sequence) of size target_size within range(n) using greedy + local search."""
        # Start with greedy construction
        sidon = set()
        for i in range(n):
            # Check if adding i keeps it a Sidon set
            valid = True
            for a in sidon:
                for b in sidon:
                    if a + b == i or a + b == i + a or a + b == i + b:
                        # More careful check: for each pair (a,b) in sidon, 
                        # and each c in sidon ∪ {i}, check no two pairs give same sum
                        pass
            # Simpler: check all pairs
            new_set = sidon | {i}
            sums = set()
            valid = True
            lst = list(new_set)
            for idx_a in range(len(lst)):
                for idx_b in range(idx_a, len(lst)):
                    s = lst[idx_a] + lst[idx_b]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                sidon.add(i)
                if len(sidon) >= target_size:
                    break
        
        if len(sidon) < target_size:
            # Try different starting points
            for start in range(1, 10):
                sidon = set()
                for i in range(start, n):
                    new_set = sidon | {i}
                    sums = set()
                    valid = True
                    lst = list(new_set)
                    for idx_a in range(len(lst)):
                        for idx_b in range(idx_a, len(lst)):
                            s = lst[idx_a] + lst[idx_b]
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                    if not valid:
                        continue
                    sidon.add(i)
                    if len(sidon) >= target_size:
                        break
                if len(sidon) >= target_size:
                    break
        
        return list(sidon)
    
    def optimize_weights_on_support(support, n, time_limit=15):
        """Optimize continuous nonnegative weights on a fixed support using smooth surrogate."""
        m = len(support)
        if m < 2:
            return None
        
        start_time = time.time()
        
        # Normalize: fix L1 = 1, so weights sum to 1
        # Objective: maximize sum(h^2) / sup|h*h|  (with L1=1)
        # Equivalently: minimize sup|h*h| / sum(h^2)
        # Use log-sum-exp for smooth max approximation
        
        def smooth_max(conv, beta=50.0):
            """Soft-max via log-sum-exp for smoothness."""
            c = conv - np.max(conv)
            return np.max(conv) + np.log(np.sum(np.exp(beta * c)) / len(c)) / beta
        
        def objective(w):
            h = np.zeros(n)
            h[support] = w
            l2_sq = np.sum(h**2)
            if l2_sq < 1e-12:
                return 1e10
            conv = np.convolve(h, h, mode='full')
            # Use smooth max as surrogate
            smax = smooth_max(conv, beta=100.0)
            return smax / l2_sq
        
        # Initial: equal weights
        w0 = np.ones(m) / m
        
        # Constraints: sum(w) = 1, w >= 0
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * m
        
        best_w = w0.copy()
        best_obj = objective(w0)
        
        # Multiple restarts with different initializations
        for restart in range(3):
            if time.time() - start_time > time_limit:
                break
            
            if restart == 0:
                w_init = w0.copy()
            else:
                # Random Dirichlet initialization
                w_init = np.random.dirichlet(np.ones(m))
            
            try:
                res = minimize(objective, w_init, method='L-BFGS-B',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_w = res.x.copy()
            except Exception:
                pass
        
        # Final evaluation with exact max
        h_final = np.zeros(n)
        h_final[support] = best_w
        final_score = compute_score(h_final)
        
        return h_final, final_score
    
    def local_search_sidon(sidon_list, n, max_iters=2000):
        """Local search: swap elements to minimize max convolution value."""
        current = list(sidon_list)
        
        # Build initial heights
        h = np.zeros(n)
        for i in current:
            h[i] = 1.0
        current_score = compute_score(h)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 10:
            improved = False
            iters += 1
            # Try swapping each element with each non-element
            for idx, elem in enumerate(current):
                for new_elem in range(n):
                    if new_elem in current:
                        continue
                    # Try the swap
                    trial = list(current)
                    trial[idx] = new_elem
                    h_trial = np.zeros(n)
                    for i in trial:
                        h_trial[i] = 1.0
                    trial_score = compute_score(h_trial)
                    if trial_score > current_score + 1e-10:
                        current = trial
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break
        
        return current, current_score
    
    # Target Sidon set sizes to try
    best_heights = np.ones(N)
    best_score = compute_score(best_heights)
    
    # Try various Sidon set sizes
    for target_size in [8, 10, 12, 14, 16, 18]:
        if time.time() > deadline - 30:
            break
        
        sidon = find_sidon_set(N, target_size)
        if len(sidon) < 4:
            continue
        
        # Local search to improve the Sidon set
        improved_sidon, sidon_score = local_search_sidon(sidon, N)
        
        if sidon_score > best_score:
            best_score = sidon_score
            h = np.zeros(N)
            for i in improved_sidon:
                h[i] = 1.0
            best_heights = h.copy()
        
        # Also try random Sidon-like sets
        for trial in range(5):
            if time.time() > deadline - 25:
                break
            # Random sparse set
            mask = np.zeros(N, dtype=bool)
            mask[random.sample(range(N), min(target_size, N))] = True
            h = mask.astype(float)
            s = compute_score(h)
            if s > best_score:
                best_score = s
                best_heights = h.copy()
    
    # Final: optimize continuous weights on best supports
    if time.time() < deadline - 15:
        # Try optimizing weights on the current best support
        support = np.where(best_heights > 0)[0]
        if len(support) > 2:
            result = optimize_weights_on_support(support, N, time_limit=10)
            if result is not None:
                h_opt, s_opt = result
                if s_opt > best_score:
                    best_score = s_opt
                    best_heights = h_opt.copy()
        
        # Try a few more random supports with weight optimization
        for trial in range(8):
            if time.time() > deadline - 10:
                break
            sz = random.choice([6, 8, 10, 12])
            sup = random.sample(range(N), sz)
            result = optimize_weights_on_support(sup, N, time_limit=5)
            if result is not None:
                h_opt, s_opt = result
                if s_opt > best_score:
                    best_score = s_opt
                    best_heights = h_opt.copy()
    
    # Ensure nonnegative and valid
    best_heights = np.maximum(best_heights, 0)
    if np.sum(best_heights) < 1e-12:
        return [1.0] * N
    
    return list(best_heights)
# EVOLVE-BLOCK-END
