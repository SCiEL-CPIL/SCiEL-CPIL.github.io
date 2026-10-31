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

def generate_sidon_sets():
    """Generate several Sidon sets of various sizes within [0, N)."""
    sidon_sets = []
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
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
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    np.random.seed(123)
    for _ in range(5000):
        k = np.random.randint(4, 11)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 50:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0, maxiter=2000):
    """
    Optimize weights h_i for i in support to minimize max convolution coefficient.
    Uses smooth max surrogate with direct weight parameterization.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform weights
    x0 = np.ones(k)
    
    def objective_and_grad(x):
        # Normalize to sum=1
        s = np.sum(x)
        if s < 1e-15:
            return 1e10, np.zeros(k)
        h_local = x / s
        
        # Full h vector
        h_full = np.zeros(N)
        h_full[support] = h_local
        
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        # Smooth max: (1/beta) * log(sum(exp(beta * conv_k)))
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        obj_val = max_c + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient: grad_h[j] = 2 * sum_{m in support} h[m] * weights[m+j]
        # Then chain rule for normalization: x -> h = x/sum(x)
        # d obj / d x_j = (grad_h_j * s - h_j * sum_i grad_h_i * s) / s^2
        # = (grad_h_j - h_j * sum_i grad_h_i) / s
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h_local[m]
            if hm > 1e-15:
                # weights[m+j] for j in [0, N-1], m in [0, N-1] => m+j in [0, 2N-2]
                valid_weights = weights[m:m+N]
                if len(valid_weights) == N:
                    grad_h += hm * valid_weights
        
        # grad_h[j] = 2 * sum_{m in support} h[m] * weights[m+j]
        grad_h *= 2.0
        
        # Chain rule for normalization
        grad_h_support = grad_h[support]
        dot_prod = np.dot(grad_h_support, h_local)
        grad_x = (grad_h_support - h_local * dot_prod) / s
        
        return obj_val, grad_x
    
    # Optimize
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        x_opt = res.x
        s = np.sum(x_opt)
        if s < 1e-15:
            h_full = np.zeros(N)
            h_full[support] = 1.0 / k
        else:
            h_full = np.zeros(N)
            h_full[support] = x_opt / s
        
        sc = compute_ratio(h_full)
        return h_full, sc
    except Exception:
        h_full = np.zeros(N)
        h_full[support] = 1.0 / k
        sc = compute_ratio(h_full)
        return h_full, sc

def local_search_swap(h, support, max_iters=50, deadline=None):
    """
    Local search: try swapping elements in support to improve the ratio.
    """
    k = len(support)
    best_h = h.copy()
    best_score = compute_ratio(best_h)
    best_support = list(support)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a new element
        for new_idx in range(N):
            if deadline is not None and time.time() > deadline:
                break
            if new_idx in best_support:
                continue
            new_support = best_support + [new_idx]
            new_support.sort()
            if len(new_support) > 15:
                continue
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0, maxiter=500)
            if sc_new > best_score + 1e-6:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing an element
        for remove_idx in list(best_support):
            if deadline is not None and time.time() > deadline:
                break
            new_support = [x for x in best_support if x != remove_idx]
            if len(new_support) < 3:
                continue
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0, maxiter=500)
            if sc_new > best_score + 1e-6:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping an element
        for old_idx in best_support:
            if deadline is not None and time.time() > deadline:
                break
            for new_idx in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                if new_idx in best_support:
                    continue
                new_support = [x for x in best_support if x != old_idx]
                new_support.append(new_idx)
                new_support.sort()
                
                h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0, maxiter=500)
                if sc_new > best_score + 1e-6:
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
    deadline = t0 + 140.0
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    best_support = [0]
    
    sidon_sets = generate_sidon_sets()
    sidon_sets.sort(key=lambda x: -len(x))
    
    # Phase 1: Optimize weights for each Sidon set with multiple beta values
    for s in sidon_sets:
        if time.time() > deadline - 40.0:
            break
        
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 40.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta, maxiter=1500)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                best_support = list(s)
    
    # Phase 2: Local search from the best solution
    if time.time() < deadline - 30.0:
        h_opt, sc_opt, support_opt = local_search_swap(best_h, best_support, max_iters=60, deadline=deadline - 20.0)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
            best_support = support_opt
    
    # Phase 3: High-precision polish with very high beta
    if time.time() < deadline - 10.0:
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 10.0:
                break
            h_polish, sc_polish = optimize_weights_for_support(best_support, beta=beta, maxiter=3000)
            if sc_polish > best_score:
                best_score = sc_polish
                best_h = h_polish.copy()
    
    return list(best_h)
