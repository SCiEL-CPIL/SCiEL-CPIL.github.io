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
    conv_offpeak = np.delete(conv, N - 1)
    conv_inf = np.max(conv_offpeak)
    if conv_inf < 1e-15:
        return h2_sq / 1e-15
    return h2_sq / conv_inf

def is_sidon_set(indices):
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
        [0, 2, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40, 45],
    ]
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            if s not in sidon_sets:
                sidon_sets.append(list(s))
    
    np.random.seed(42)
    seen = set(tuple(sorted(s)) for s in sidon_sets)
    for _ in range(8000):
        k = np.random.randint(4, 13)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            key = tuple(indices)
            if key not in seen:
                seen.add(key)
                sidon_sets.append(indices)
                if len(sidon_sets) > 60:
                    break
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0, deadline=None):
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    support_arr = np.array(support)
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def build_conv_matrix():
        """Precompute: for each lag l, which pairs (i,j) in support give i+j = N-1+l"""
        # conv index = i + j + (N-1), so lag = i + j - (N-1)
        # We need conv[k] for k != N-1, i.e., lag != 0
        # conv[k] = sum_{i+j=k} h_i * h_j
        # For support indices, compute all pair sums
        conv_coeffs = {}
        for ii in range(k):
            for jj in range(k):
                c = support_arr[ii] + support_arr[jj]
                if c == N - 1:
                    continue  # skip lag 0
                conv_coeffs.setdefault(c, []).append((ii, jj))
        return conv_coeffs
    
    conv_coeffs = build_conv_matrix()
    conv_lags = sorted(conv_coeffs.keys())
    n_conv = len(conv_lags)
    
    if n_conv == 0:
        h = np.zeros(N)
        h[support] = 1.0 / k
        return h, compute_ratio(h)
    
    # For each conv lag, store the pairs
    conv_pairs = []
    for lag in conv_lags:
        pairs = np.array(conv_coeffs[lag])
        conv_pairs.append(pairs)
    
    def objective_and_grad(x):
        h = softmax(x)
        
        # Compute all off-peak conv values
        conv_vals = np.zeros(n_conv)
        for idx, pairs in enumerate(conv_pairs):
            conv_vals[idx] = np.sum(h[pairs[:, 0]] * h[pairs[:, 1]])
        
        max_c = np.max(conv_vals)
        if max_c < 1e-15:
            max_c = 1e-15
        
        shifted = conv_vals - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        h2_sq = np.sum(h**2)
        if h2_sq < 1e-15:
            h2_sq = 1e-15
            
        obj_val = (1.0 / beta) * np.log(sum_exp) + max_c - np.log(h2_sq)
        
        # Gradient: d/dx of obj
        # d/dh_j of (1/beta)*log(sum exp(beta*conv_k)) = (1/beta)*sum_k weights_k * d conv_k/dh_j
        # d conv_k / dh_j = sum_{i: i+j=k} h_i + sum_{i: j+i=k} h_i (for j != i) or 2*h_j (for j=i)
        # Actually conv_k = sum_{i+j=k} h_i h_j, so d conv_k/dh_j = sum_{i: i+j=k} h_i (counting multiplicity)
        
        grad_h = np.zeros(k)
        for idx, pairs in enumerate(conv_pairs):
            w = weights[idx]
            for ii, jj in pairs:
                grad_h[ii] += w * h[jj]
                grad_h[jj] += w * h[ii]
        
        grad_h *= (1.0 / beta)
        
        # d/dh_j of -log(h2_sq) = -2*h_j / h2_sq
        grad_h -= 2.0 * h / h2_sq
        
        # Chain rule for softmax
        dot_prod = np.dot(grad_h, h)
        grad_x = h * (grad_h - dot_prod)
        
        return obj_val, grad_x
    
    best_local_h = np.zeros(N)
    best_local_score = -np.inf
    best_local_x = np.zeros(k)
    
    np.random.seed(42 + k * 7)
    n_restarts = min(6, max(2, 70 - 6 * len(support)))
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        if restart == 0:
            x_curr = np.zeros(k)
        elif restart == 1:
            x_curr = np.random.randn(k) * 0.5
        else:
            x_curr = np.random.randn(k) * 2.0
            
        try:
            res = minimize(objective_and_grad, x_curr, method='L-BFGS-B',
                           jac=True, options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
            x_opt = res.x
            h_opt_local = softmax(x_opt)
            
            h_full = np.zeros(N)
            h_full[support] = h_opt_local
            sc = compute_ratio(h_full)
            
            if sc > best_local_score:
                best_local_score = sc
                best_local_h = h_full.copy()
                best_local_x = x_opt.copy()
        except Exception:
            continue
    
    # Polish with higher beta
    if best_local_score > -np.inf and deadline is not None:
        remaining = deadline - time.time()
        if remaining > 1.0:
            for beta_polish in [200.0, 500.0]:
                if time.time() > deadline:
                    break
                try:
                    res = minimize(objective_and_grad.__wrapped__ if hasattr(objective_and_grad, '__wrapped__') else objective_and_grad, 
                                   best_local_x, method='L-BFGS-B',
                                   jac=True, options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
                    # Need to re-create with new beta
                    def obj_polish(x, bp=beta_polish):
                        h = softmax(x)
                        conv_vals = np.zeros(n_conv)
                        for idx, pairs in enumerate(conv_pairs):
                            conv_vals[idx] = np.sum(h[pairs[:, 0]] * h[pairs[:, 1]])
                        max_c = np.max(conv_vals)
                        if max_c < 1e-15:
                            max_c = 1e-15
                        shifted = conv_vals - max_c
                        exp_shifted = np.exp(bp * shifted)
                        sum_exp = np.sum(exp_shifted)
                        weights = exp_shifted / sum_exp
                        h2_sq = np.sum(h**2)
                        if h2_sq < 1e-15:
                            h2_sq = 1e-15
                        obj_val = (1.0 / bp) * np.log(sum_exp) + max_c - np.log(h2_sq)
                        grad_h = np.zeros(k)
                        for idx, pairs in enumerate(conv_pairs):
                            w = weights[idx]
                            for ii, jj in pairs:
                                grad_h[ii] += w * h[jj]
                                grad_h[jj] += w * h[ii]
                        grad_h *= (1.0 / bp)
                        grad_h -= 2.0 * h / h2_sq
                        dot_prod = np.dot(grad_h, h)
                        grad_x = h * (grad_h - dot_prod)
                        return obj_val, grad_x
                    
                    res = minimize(obj_polish, best_local_x, method='L-BFGS-B',
                                   jac=True, options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
                    x_opt = res.x
                    h_opt_local = softmax(x_opt)
                    h_full = np.zeros(N)
                    h_full[support] = h_opt_local
                    sc = compute_ratio(h_full)
                    
                    if sc > best_local_score:
                        best_local_score = sc
                        best_local_h = h_full.copy()
                        best_local_x = x_opt.copy()
                except Exception:
                    pass
    
    if best_local_score == -np.inf:
        h_full = np.zeros(N)
        h_full[support] = 1.0 / k
        best_local_score = compute_ratio(h_full)
        best_local_h = h_full
    
    return best_local_h, best_local_score

def local_search_swap(h, support, max_iters=60, deadline=None):
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
            new_support = sorted(best_support + [new_idx])
            if len(new_support) > 14:
                continue
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=60.0, deadline=deadline)
            if sc_new > best_score + 1e-5:
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
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=60.0, deadline=deadline)
            if sc_new > best_score + 1e-5:
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
                
                h_new, sc_new = optimize_weights_for_support(new_support, beta=60.0, deadline=deadline)
                if sc_new > best_score + 1e-5:
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
    
    # Phase 1: Try all Sidon sets with moderate beta
    for s in sidon_sets:
        if time.time() > deadline - 40.0:
            break
        for beta in [40.0, 80.0, 150.0]:
            if time.time() > deadline - 40.0:
                break
            h_opt, sc = optimize_weights_for_support(s, beta=beta, deadline=deadline - 40.0)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                best_support = list(s)
    
    # Phase 2: Local search on support
    if time.time() < deadline - 30.0:
        h_opt, sc_opt, support_opt = local_search_swap(best_h, best_support, max_iters=50, deadline=deadline - 20.0)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
            best_support = support_opt
    
    # Phase 3: Final polish with high beta
    if time.time() < deadline - 5.0:
        for beta in [200.0, 400.0, 800.0]:
            if time.time() > deadline - 3.0:
                break
            h_polish, sc_polish = optimize_weights_for_support(best_support, beta=beta, deadline=deadline - 2.0)
            if sc_polish > best_score:
                best_score = sc_polish
                best_h = h_polish.copy()
    
    return list(best_h)
