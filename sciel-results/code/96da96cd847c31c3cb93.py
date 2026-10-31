import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_good_sidon_sets(max_size=15, max_sets=100):
    """Find good Sidon sets using a greedy approach with backtracking."""
    sidon_sets = []
    
    # Known good Sidon sets (from literature and computation)
    known_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40, 45, 48],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in known_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    # Try to find more Sidon sets with random restarts
    np.random.seed(42)
    for _ in range(20000):
        k = np.random.randint(5, min(max_size + 1, N))
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            # Check if this set is new
            is_new = True
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    is_new = False
                    break
            if is_new:
                sidon_sets.append(indices)
                if len(sidon_sets) >= max_sets:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=50.0, maxiter=2000, ftol=1e-14, time_budget=10.0):
    """
    Optimize weights h_i for i in support to maximize the ratio.
    Uses a smooth surrogate for the max convolution coefficient.
    Parameterization: h = softmax(x) ensures positivity and sum=1.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    t_start = time.time()
    
    # Initial guess: uniform weights in log-space
    x0 = np.zeros(k)
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad(x):
        # Time check inside objective for early termination if needed
        if time.time() - t_start > time_budget:
            raise TimeoutError()
            
        h = softmax(x)
        # Full h vector
        h_full = np.zeros(N)
        h_full[support] = h
        
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        # Objective: (1/beta) * log(sum(exp(beta * conv_k)))
        # = max_c + (1/beta) * log(sum(exp(beta * shifted)))
        obj_val = max_c + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient computation (vectorized)
        grad_h = np.zeros(N)
        for m in support:
            hm = h_full[m]
            if hm > 1e-15:
                end = min(m + N, 2 * N - 1)
                grad_h[:N] += 2.0 * hm * weights[m:end]
        
        # Chain rule: h = softmax(x)
        grad_h_support = grad_h[support]
        dot_prod = np.dot(grad_h_support, h)
        grad_x = h * (grad_h_support - dot_prod)
        
        return obj_val, grad_x
    
    # Optimize
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-12})
        x_opt = res.x
        h_opt_local = softmax(x_opt)
        
        h_full = np.zeros(N)
        h_full[support] = h_opt_local
        sc = compute_ratio(h_full)
        return h_full, sc
    except (Exception, TimeoutError):
        # Fallback to uniform
        h_full = np.zeros(N)
        h_full[support] = 1.0 / k
        sc = compute_ratio(h_full)
        return h_full, sc

def local_search_swap(h, support, max_iters=50, time_limit=None):
    """
    Local search: try swapping elements in support to improve the ratio.
    """
    k = len(support)
    best_h = h.copy()
    best_score = compute_ratio(best_h)
    best_support = list(support)
    
    t0 = time.time()
    
    for iteration in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
            
        improved = False
        
        # Try adding a new element
        for new_idx in range(N):
            if new_idx in best_support:
                continue
            new_support = best_support + [new_idx]
            new_support.sort()
            if len(new_support) > 18:  # Limit size
                continue
            
            # Optimize weights for new support
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0, maxiter=500, time_budget=2.0)
            if sc_new > best_score + 1e-8:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved or (time_limit is not None and time.time() > time_limit):
            continue
        
        # Try removing an element
        for remove_idx in list(best_support):
            new_support = [x for x in best_support if x != remove_idx]
            if len(new_support) < 3:
                continue
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0, maxiter=500, time_budget=2.0)
            if sc_new > best_score + 1e-8:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved or (time_limit is not None and time.time() > time_limit):
            continue
        
        # Try swapping an element
        for old_idx in best_support:
            for new_idx in range(N):
                if new_idx in best_support:
                    continue
                new_support = [x for x in best_support if x != old_idx]
                new_support.append(new_idx)
                new_support.sort()
                
                h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0, maxiter=500, time_budget=2.0)
                if sc_new > best_score + 1e-8:
                    best_h = h_new
                    best_score = sc_new
                    best_support = new_support
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_h, best_score, best_support

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    best_support = [0]
    
    # Find good Sidon sets
    sidon_sets = find_good_sidon_sets(max_size=15, max_sets=150)
    
    # Sort by size descending, then by some heuristic
    sidon_sets.sort(key=lambda x: (-len(x), x[0] if x else 0))
    
    # Phase 1: Optimize weights for each Sidon set with multiple betas
    for s in sidon_sets:
        if time.time() > deadline - 30.0:
            break
        
        # Use a range of betas
        for beta in [30.0, 80.0, 200.0]:
            if time.time() > deadline - 30.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta, maxiter=1500, time_budget=5.0)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                best_support = list(s)
    
    # Phase 2: Local search from the best solution
    if time.time() < deadline - 40.0:
        h_opt, sc_opt, support_opt = local_search_swap(best_h, best_support, max_iters=40, time_limit=deadline - 40.0)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
            best_support = support_opt
    
    # Phase 3: High-precision polish with very high beta
    if time.time() < deadline - 15.0:
        for beta in [300.0, 600.0, 1200.0]:
            if time.time() > deadline - 15.0:
                break
            h_polish, sc_polish = optimize_weights_for_support(best_support, beta=beta, maxiter=3000, ftol=1e-15, time_budget=10.0)
            if sc_polish > best_score:
                best_score = sc_polish
                best_h = h_polish.copy()
    
    # Phase 4: Final local search with high precision
    if time.time() < deadline - 5.0:
        h_opt, sc_opt, support_opt = local_search_swap(best_h, best_support, max_iters=20, time_limit=deadline - 5.0)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
            best_support = support_opt
    
    return list(best_h)
