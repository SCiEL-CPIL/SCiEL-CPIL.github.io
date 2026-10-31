import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
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

def greedy_sidon_search(indices, deadline):
    """Greedy heuristic with random restarts to find a large Sidon subset of indices.
    Returns the best Sidon set found."""
    best_sidon = set()
    for restart in range(50):
        if time.time() > deadline - 20.0:
            break
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        current = set()
        for idx in shuffled:
            test_set = list(current) + [idx]
            if is_sidon_set(test_set):
                current.add(idx)
        if len(current) > len(best_sidon):
            best_sidon = set(current)
    return list(best_sidon)

def smooth_max_surrogate(conv, temperature=0.1):
    """Log-sum-exp approximation to max for smooth optimization."""
    scaled = conv / temperature
    max_val = np.max(scaled)
    if max_val > 700:
        max_val = 700
    return temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))

def optimize_weights_logspace(indices, h_start, deadline, use_smooth=False, temperature=0.05):
    """Optimize weights on fixed support using log-space parametrization."""
    k = len(indices)
    if k == 0:
        return np.zeros(N)
    
    h_init = h_start[indices].copy()
    h_init = np.maximum(h_init, 1e-10)
    x_init = np.log(h_init)
    
    def objective(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        h = exp_x / np.sum(exp_x)
        
        h2_sq = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        
        if use_smooth:
            conv_inf = smooth_max_surrogate(conv, temperature)
        else:
            conv_inf = np.max(conv)
        
        if conv_inf < 1e-15:
            return 1e10
        return -h2_sq / conv_inf
    
    res = minimize(objective, x_init, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    x_opt = res.x
    x_shifted = x_opt - np.max(x_opt)
    exp_x = np.exp(x_shifted)
    h_opt = exp_x / np.sum(exp_x)
    
    h_full = np.zeros(N)
    h_full[indices] = h_opt
    return h_full

def gradient_polish(h, deadline, max_iter=200):
    """Gradient-based polish using L-BFGS-B on simplex-constrained weights."""
    h = h.copy()
    indices = np.where(h > 1e-10)[0]
    if len(indices) < 2:
        return h
    
    k = len(indices)
    h_sub = h[indices].copy()
    h_sub = np.maximum(h_sub, 1e-10)
    h_sub = h_sub / np.sum(h_sub)
    
    def objective_log(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        h = exp_x / np.sum(exp_x)
        
        h2_sq = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        conv_inf = np.max(conv)
        
        if conv_inf < 1e-15:
            return 1e10
        return -h2_sq / conv_inf
    
    def gradient_log(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        h = exp_x / np.sum(exp_x)
        
        h2_sq = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        conv_inf = np.max(conv)
        
        if conv_inf < 1e-15:
            return np.zeros(k)
        
        ratio = h2_sq / conv_inf
        
        # Gradient of h2_sq = sum(h_i^2) w.r.t. log weights
        # h_i = exp(x_i) / sum(exp(x_j))
        # dh_i/dx_j = h_i * (delta_ij - h_j)
        # d(h2_sq)/dx_j = 2 * sum_i h_i * dh_i/dx_j = 2 * h_j * (h2_sq - h_j * sum_i h_i^2 / h_j)
        # Actually: d(h2_sq)/dx_j = 2 * h_j * (h2_sq - h_j)
        
        # Gradient of conv_inf w.r.t. h: need to find which index achieves max
        max_idx = np.argmax(conv)
        
        # conv[m] = sum_i h[i] * h[m-i] (with appropriate indexing for fftconvolve)
        # d(conv[m])/d(h[j]) = h[m-j] + h[j] (if m-j is valid)
        # For full mode: conv has length 2*N-1, conv[m] = sum_{i+j=m} h[i]*h[j]
        # d(conv[m])/d(h[j]) = sum_i [delta_{i+j,m} * h[i] + delta_{i+j,m} * h[j]]
        #                     = h[m-j] + h[j] (when 0 <= m-j < N)
        
        grad_conv_inf = np.zeros(k)
        m = max_idx
        for j in range(k):
            # h is defined on indices, but we're working in local coordinates
            # Need to map local index j to global index
            global_j = indices[j]
            other = m - global_j
            if 0 <= other < N:
                # Check if 'other' is in our support
                other_pos = np.where(indices == other)[0]
                if len(other_pos) > 0:
                    grad_conv_inf[j] += h[other_pos[0]]
            # Also the symmetric term
            grad_conv_inf[j] += h[j]
        
        # Combine gradients
        # d(-ratio)/dx_j = -(1/conv_inf) * d(h2_sq)/dx_j + (h2_sq/conv_inf^2) * d(conv_inf)/dx_j
        grad_h2 = 2 * h * (h2_sq - h)
        grad = -(grad_h2 / conv_inf) + (h2_sq / conv_inf**2) * grad_conv_inf
        
        return grad
    
    x_init = np.log(h_sub)
    
    res = minimize(objective_log, x_init, method='L-BFGS-B',
                   jac=gradient_log,
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
    
    x_opt = res.x
    x_shifted = x_opt - np.max(x_opt)
    exp_x = np.exp(x_shifted)
    h_opt = exp_x / np.sum(exp_x)
    
    h[indices] = h_opt
    return h

def local_search_polish(h, deadline):
    """Enhanced local search with patience and more perturbations."""
    h = h.copy()
    improved = True
    max_iter = 200
    patience = 0
    
    for _ in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        if not improved and patience > 5:
            break
        improved = False
        current_score = compute_ratio(h)
        
        nonzero_idx = set(np.where(h > 1e-10)[0])
        zero_idx = [i for i in range(N) if i not in nonzero_idx]
        
        for idx in zero_idx[:10]:
            if time.time() > deadline - 10.0:
                break
            for w in [0.01, 0.05, 0.1, 0.2, 0.5]:
                h_test = h.copy()
                h_test[idx] = w
                sc = compute_ratio(h_test)
                if sc > current_score + 1e-10:
                    h = h_test
                    current_score = sc
                    improved = True
                    patience = 0
                    break
            if improved:
                break
        
        if not improved:
            nonzero_list = list(nonzero_idx)
            for idx in nonzero_list:
                if time.time() > deadline - 10.0:
                    break
                h_test = h.copy()
                h_test[idx] = 0.0
                sc = compute_ratio(h_test)
                if sc > current_score + 1e-10:
                    h = h_test
                    current_score = sc
                    improved = True
                    patience = 0
                    break
        
        if not improved:
            nonzero_list = list(nonzero_idx)
            for idx in nonzero_list:
                if time.time() > deadline - 10.0:
                    break
                for factor in [0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]:
                    h_test = h.copy()
                    h_test[idx] *= factor
                    sc = compute_ratio(h_test)
                    if sc > current_score + 1e-10:
                        h = h_test
                        current_score = sc
                        improved = True
                        patience = 0
                        break
                if improved:
                    break
        
        if not improved:
            patience += 1
    
    return h

def high_precision_polish(h, deadline):
    """Final high-precision polish with tight tolerances."""
    h = h.copy()
    indices = np.where(h > 1e-10)[0]
    if len(indices) == 0:
        return h
    
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        current_score = compute_ratio(h)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
            for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]:
                h_test = h.copy()
                h_test[idx] = max(0, h_test[idx] + delta)
                sc = compute_ratio(h_test)
                if sc > current_score + 1e-12:
                    h = h_test
                    current_score = sc
                    break
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
            for factor in [0.95, 0.98, 1.02, 1.05]:
                h_test = h.copy()
                h_test[idx] *= factor
                sc = compute_ratio(h_test)
                if sc > current_score + 1e-12:
                    h = h_test
                    current_score = sc
                    break
    
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []

    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))

    np.random.seed(42)

    all_indices = list(range(N))
    
    best_sidon = []
    for restart in range(20):
        if time.time() > deadline - 60.0:
            break
        sidon_set = greedy_sidon_search(all_indices, deadline)
        if len(sidon_set) > len(best_sidon):
            best_sidon = sidon_set
    
    known_sidon_patterns = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 3, 7, 15, 31],
        [0, 2, 5, 11, 23],
        [0, 1, 4, 9, 16, 25, 36, 49],
        [0, 1, 5, 11, 25, 26, 31, 37, 43, 49],
        [0, 2, 3, 7, 12, 18, 25, 31, 38, 44, 49],
    ]
    
    all_sidon_candidates = [best_sidon] + known_sidon_patterns
    
    for sidon_set in all_sidon_candidates:
        if time.time() > deadline - 50.0:
            break
        if len(sidon_set) < 2:
            continue
        h = np.zeros(N)
        for idx in sidon_set:
            if 0 <= idx < N:
                h[idx] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)
    
    num_sparse_tries = 800
    for _ in range(num_sparse_tries):
        if time.time() > deadline - 40.0:
            break
        
        k = np.random.randint(3, 15)
        indices = np.random.choice(N, k, replace=False)
        weights = np.random.rand(k)
        weights = weights ** np.random.uniform(0.5, 2.0)
        h = np.zeros(N)
        h[indices] = weights
        
        sc = compute_ratio(h)
        update_best(h, sc)

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = candidates[:20]
    
    for i, (sc, h_start) in enumerate(top_k):
        if time.time() > deadline - 25.0:
            break
        
        indices = np.where(h_start > 1e-10)[0]
        if len(indices) == 0:
            continue
            
        for use_smooth in [False, True]:
            if time.time() > deadline - 25.0:
                break
            
            h_opt = optimize_weights_logspace(indices, h_start, deadline,
                                              use_smooth=use_smooth, temperature=0.05)
            
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc_opt = compute_ratio(h_opt)
                update_best(h_opt, sc_opt)
        
        # Add gradient polish step
        if time.time() > deadline - 25.0:
            continue
        
        h_grad = gradient_polish(h_opt, deadline, max_iter=100)
        sc_grad = compute_ratio(h_grad)
        update_best(h_grad, sc_grad)
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_polish = candidates[:8]
    
    for i, (sc, h_start) in enumerate(top_polish):
        if time.time() > deadline - 10.0:
            break
        
        h_polished = local_search_polish(h_start, deadline)
        sc_polished = compute_ratio(h_polished)
        update_best(h_polished, sc_polished)
        
        # Additional gradient polish after local search
        if time.time() > deadline - 8.0:
            continue
        
        h_grad2 = gradient_polish(h_polished, deadline, max_iter=50)
        sc_grad2 = compute_ratio(h_grad2)
        update_best(h_grad2, sc_grad2)
    
    if time.time() < deadline - 5.0:
        h_final = high_precision_polish(best_h, deadline)
        sc_final = compute_ratio(h_final)
        update_best(h_final, sc_final)
    
    # Final gradient polish
    if time.time() < deadline - 2.0:
        h_last = gradient_polish(best_h, deadline, max_iter=50)
        sc_last = compute_ratio(h_last)
        update_best(h_last, sc_last)

    return list(best_h)
