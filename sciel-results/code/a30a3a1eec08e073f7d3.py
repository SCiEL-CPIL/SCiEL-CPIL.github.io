import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_MARGIN = 15.0

def compute_ratio_exact(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def smooth_max(x, temp=1e-3):
    x_max = np.max(x)
    return x_max + temp * np.log(np.sum(np.exp((x - x_max) / temp)))

def optimize_weights_for_support(support, t0, deadline):
    if len(support) < 2:
        return 0.0, np.zeros(N)
        
    n_vars = len(support)
    
    # Initial guess: uniform
    x0 = np.zeros(n_vars)
    bounds = [(-10, 10)] * n_vars
    
    def neg_ratio_smooth_weights(params, support, temp=1e-3):
        w = np.exp(params)
        w = w / np.sum(w)
        
        # Construct full vector
        h = np.zeros(N)
        h[support] = w
        
        h2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        conv_soft_max = smooth_max(conv, temp)
        
        if conv_soft_max < 1e-15:
            return 1e10
        return -(h2_sq / conv_soft_max)
        
    try:
        res = minimize(neg_ratio_smooth_weights, x0, args=(support, 1e-3),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-10})
        
        if res.success or res.fun < 1e10:
            w_opt = np.exp(res.x)
            w_opt = w_opt / np.sum(w_opt)
            
            h_final = np.zeros(N)
            h_final[support] = w_opt
            
            score = compute_ratio_exact(h_final)
            return score, h_final
    except Exception:
        pass
        
    return 0.0, np.zeros(N)

def find_sidon_set_greedy(n, order, max_size=None):
    if max_size is None:
        max_size = 20  # Cap to avoid huge sets that might be hard to optimize or not optimal
    A = []
    used_sums = set()
    for x in order:
        if len(A) >= max_size:
            break
        valid = True
        new_sums = []
        for a in A:
            s = a + x
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if 2 * x in used_sums:
            valid = False
        if valid:
            A.append(x)
            used_sums.update(new_sums)
            used_sums.add(2 * x)
    return A

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    
    best_score = 0.0
    best_h = np.zeros(N)
    
    # Phase 1: Generate candidate supports
    supports = []
    seen = set()
    
    def add_support(S):
        if len(S) < 2:
            return
        S_sorted = sorted(S)
        t = tuple(S_sorted)
        if t not in seen:
            seen.add(t)
            supports.append(S_sorted)

    # 1. Greedy Sidon sets with various orderings
    # Standard order
    S = find_sidon_set_greedy(N, list(range(N)))
    add_support(S)
    
    # Random restarts for Sidon sets
    rng = np.random.default_rng(42)
    for _ in range(50):
        order = list(range(N))
        rng.shuffle(order)
        S = find_sidon_set_greedy(N, order)
        add_support(S)
        
    # 2. Specific patterns
    # Powers of 2
    add_support([0, 1, 2, 4, 8, 16, 32])
    
    # Small random sparse sets
    for _ in range(50):
        k = rng.integers(3, 12)
        S = sorted(rng.choice(N, size=k, replace=False).tolist())
        add_support(S)
        
    # 3. Centered clusters?
    # Try a few fixed patterns around the center
    center = N // 2
    for width in [3, 5, 7]:
        for step in [1, 2]:
            S = list(range(center - width//2, center + width//2 + 1, step))
            S = [s for s in S if 0 <= s < N]
            add_support(S)

    # Phase 2: Optimize weights for each support
    # Sort supports by size descending? Or just iterate?
    # Larger Sidon sets often give better base scores, but optimization is harder.
    # Let's just iterate through all of them.
    for S in supports:
        if time.time() > deadline - 5:
            break
            
        score, h = optimize_weights_for_support(S, t0, deadline)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
    # Phase 3: Final polish on the best support found
    # Identify the support of the best_h
    support_best = np.nonzero(best_h)[0].tolist()
    if len(support_best) > 0:
        # Run a more precise optimization on this support
        # Use smaller temp for smooth max
        def neg_ratio_smooth_final(params, support):
            w = np.exp(params)
            w = w / np.sum(w)
            h = np.zeros(N)
            h[support] = w
            h2_sq = np.sum(h**2)
            conv = np.convolve(h, h, mode='full')
            conv_soft_max = smooth_max(conv, 1e-5)
            if conv_soft_max < 1e-15:
                return 1e10
            return -(h2_sq / conv_soft_max)
            
        n_vars = len(support_best)
        x0 = np.log(np.maximum(best_h[support_best], 1e-10))
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = minimize(neg_ratio_smooth_final, x0, args=(support_best,),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12})
            w_opt = np.exp(res.x)
            w_opt = w_opt / np.sum(w_opt)
            h_final = np.zeros(N)
            h_final[support_best] = w_opt
            score_final = compute_ratio_exact(h_final)
            if score_final > best_score:
                best_score = score_final
                best_h = h_final.copy()
        except Exception:
            pass

    return list(best_h)
