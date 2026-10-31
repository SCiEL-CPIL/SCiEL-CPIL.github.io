import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50
DEADLINE_MARGIN = 10.0

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

def neg_ratio_smooth_weights(params, support, temp=1e-3):
    """
    Optimize weights for a fixed support.
    params: weights for indices in support.
    support: list of indices in 0..N-1.
    """
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

def optimize_weights_for_support(support, t0, deadline):
    if len(support) < 2:
        return 0.0, np.zeros(N)
        
    n_vars = len(support)
    # Initial guess: uniform
    x0 = np.zeros(n_vars)
    
    # Bounds: log-space, so (-inf, inf). But to prevent overflow, maybe (-10, 10)?
    # exp(10) is huge, but we normalize.
    # Actually, we can just use bounds (0, inf) on weights directly with SLSQP?
    # SLSQP handles bounds well.
    # Let's try SLSQP on weights directly to avoid log-space issues with zeros.
    
    def objective(weights):
        h = np.zeros(N)
        h[support] = weights
        l1 = np.sum(weights)
        if l1 < 1e-15:
            return 1e10
        h_norm = h / l1
        h2_sq = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        conv_inf = np.max(conv)
        if conv_inf < 1e-15:
            return 1e10
        return -(h2_sq / conv_inf)
        
    # SLSQP might be slow or fail on non-smooth max.
    # Let's stick to log-space with L-BFGS-B using smooth max, but constrain the support.
    # The previous issue was that L-BFGS-B on the full vector allowed other elements to become non-zero.
    # Here, we ONLY optimize the weights on the support. The other elements are fixed to 0.
    # This preserves the sparsity.
    
    x0 = np.zeros(n_vars)
    bounds = [(-10, 10)] * n_vars
    
    try:
        res = minimize(neg_ratio_smooth_weights, x0, args=(support, 1e-3),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-12})
        
        if res.success or res.fun < 1e10:
            w_opt = np.exp(res.x)
            w_opt = w_opt / np.sum(w_opt)
            
            h_final = np.zeros(N)
            h_final[support] = w_opt
            
            # Refine with exact ratio if time permits?
            # Or just use the smooth result.
            score = compute_ratio_exact(h_final)
            return score, h_final
    except Exception:
        pass
        
    return 0.0, np.zeros(N)

def find_sidon_set_greedy(n, start_idx=0, step=1, max_size=10):
    A = []
    used_sums = set()
    for x in range(start_idx, n, step):
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

def find_sidon_set_random(n, rng, max_size=10):
    """Find a Sidon set using random greedy approach."""
    A = []
    used_sums = set()
    candidates = list(range(n))
    rng.shuffle(candidates)
    
    for x in candidates:
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
    return sorted(A)

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    
    best_score = 0.0
    best_h = np.zeros(N)
    
    # Phase 1: Generate candidate supports
    supports = []
    
    # 1. Greedy Sidon sets with more parameters
    for start in range(0, 8):
        for step in [1, 2, 3, 4]:
            S = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(S) >= 3:
                supports.append(S)
                
    # 2. Random Sidon sets
    rng = np.random.default_rng(42)
    for _ in range(100):
        k = rng.integers(4, 15)
        S = find_sidon_set_random(N, rng, max_size=k)
        if len(S) >= 3:
            supports.append(S)
        
    # 3. Specific known good patterns
    supports.append([0, 1, 2, 4, 8, 16, 32])
    supports.append([0, 1, 3, 7, 15, 31])
    supports.append([0, 1, 4, 10, 22, 44])
    
    # 4. Random sparse sets
    for _ in range(80):
        k = rng.integers(3, 14)
        S = sorted(rng.choice(N, size=k, replace=False).tolist())
        supports.append(S)
    
    # Remove duplicates (as tuples)
    unique_supports = []
    seen = set()
    for S in supports:
        t = tuple(S)
        if t not in seen:
            seen.add(t)
            unique_supports.append(S)
            
    # Phase 2: Optimize weights for each support
    for S in unique_supports:
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
            conv_soft_max = smooth_max(conv, 1e-6)
            if conv_soft_max < 1e-15:
                return 1e10
            return -(h2_sq / conv_soft_max)
            
        n_vars = len(support_best)
        x0 = np.log(np.maximum(best_h[support_best], 1e-10))
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = minimize(neg_ratio_smooth_final, x0, args=(support_best,),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14})
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
