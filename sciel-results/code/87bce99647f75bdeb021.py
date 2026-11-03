import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def compute_score(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    h is a numpy array of non-negative floats.
    """
    h = np.asarray(h)
    if np.any(h < -1e-12):
        return -1.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    conv = np.convolve(h, h, mode='full')
    l_inf = np.max(conv)
    
    if l_inf < 1e-12:
        return -1.0
        
    return l2_sq / (l1 * l_inf)

def optimize_weights_for_support(A, h0_weights=None, time_limit=2.0):
    """
    Optimize weights for a fixed support A.
    Variables: w (length |A|)
    Objective: Maximize sum(w^2) / max_k (sum_{i,j in A, i+j=k} w_i w_j)
    Subject to: sum(w) = 1, w >= 0
    """
    if not A:
        return np.zeros(N)
        
    A = np.array(A)
    m = len(A)
    
    # Initial weights
    if h0_weights is None:
        w = np.ones(m) / m
    else:
        w = np.array(h0_weights)
        s = np.sum(w)
        if s < 1e-12:
            w = np.ones(m) / m
        else:
            w = w / s
            
    # Precompute pairs for the support
    # We need to evaluate the max convolution peak for given w.
    # Conv peak at k is sum_{i+j=k} w_i w_j.
    # Instead of full convolve every time (expensive), we can use the fact that
    # we only care about the max.
    
    # For SLSQP, we need objective and gradient.
    # Objective: - (sum(w^2) / max_peak(w))
    
    def objective(w):
        w = np.maximum(w, 0)
        s = np.sum(w)
        if s < 1e-12:
            return 1e6
        w_norm = w / s
        
        # Compute convolution peak
        # Efficient way:
        # conv = np.convolve(w, w, mode='full') is O(m^2).
        # But we need the max.
        # Let's just do it. m is small (< 20 usually).
        conv = np.convolve(w_norm, w_norm, mode='full')
        peak = np.max(conv)
        
        if peak < 1e-12:
            return 1e6 # Avoid div by zero, effectively infinite score
            
        l2_sq = np.sum(w_norm**2)
        return -(l2_sq / peak)
        
    def gradient(w):
        # Numerical gradient is safer for this complex objective
        # Or analytic?
        # Let's use numerical for reliability, m is small.
        grad = np.zeros(m)
        eps = 1e-5
        f0 = objective(w)
        for i in range(m):
            w_pert = w.copy()
            w_pert[i] += eps
            grad[i] = (objective(w_pert) - f0) / eps
        return grad

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    bounds = [(0, None) for _ in range(m)]
    
    start_time = time.time()
    best_w = w.copy()
    best_val = objective(w)
    
    try:
        res = minimize(objective, w, method='SLSQP', jac=gradient, bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
        if res.success:
            w_res = res.x
            w_res = np.maximum(w_res, 0)
            s = np.sum(w_res)
            if s > 1e-12:
                w_res = w_res / s
                val_res = objective(w_res)
                if val_res < best_val:
                    best_w = w_res
                    best_val = val_res
    except Exception:
        pass
        
    # Construct full vector
    h_full = np.zeros(N)
    h_full[A] = best_w
    return h_full

def generate_sidon_set(n, target_size):
    """
    Greedy construction of a Sidon-like set in [0, n-1].
    """
    A = set()
    sums = set()
    
    for x in range(n):
        new_sums = []
        conflict = False
        for y in A:
            s = x + y
            if s in sums:
                conflict = True
                break
            new_sums.append(s)
        if not conflict:
            s2 = 2 * x
            if s2 in sums:
                conflict = True
            else:
                new_sums.append(s2)
        
        if not conflict:
            A.add(x)
            sums.update(new_sums)
            
        if len(A) >= target_size:
            break
            
    return sorted(A)

def construct():
    deadline = time.time() + 140.0
    best_h = None
    best_score = -1.0
    
    # Candidate 1: Uniform
    h_uni = np.ones(N)
    score_uni = compute_score(h_uni)
    if score_uni > best_score:
        best_score = score_uni
        best_h = h_uni.copy()
        
    # Candidate 2: Sidon sets with optimized weights
    # Try different sizes
    for size in [5, 7, 8, 10, 12]:
        if time.time() > deadline:
            break
        A = generate_sidon_set(N, size)
        if len(A) < size:
            # Try to fill or skip
            if len(A) == 0:
                continue
        h_opt = optimize_weights_for_support(A, time_limit=2.0)
        score_opt = compute_score(h_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # Candidate 3: Random sparse supports with optimized weights
    for _ in range(15):
        if time.time() > deadline:
            break
        density = random.uniform(0.1, 0.3)
        mask = np.random.rand(N) < density
        A = np.where(mask)[0].tolist()
        if len(A) < 2:
            continue
        h_opt = optimize_weights_for_support(A, time_limit=1.0)
        score_opt = compute_score(h_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # Candidate 4: Dense random with gradient ascent (fast)
    def fast_grad_ascent(h, steps=500):
        h = np.array(h, dtype=float)
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-12: return h
        h = h / s
        
        eta = 0.05
        best_h_local = h.copy()
        best_score_local = compute_score(h)
        
        for _ in range(steps):
            conv = np.convolve(h, h, mode='full')
            max_idx = np.argmax(conv)
            M = conv[max_idx]
            if M < 1e-12: break
            Q = np.sum(h**2)
            
            grad_Q = 2.0 * h
            grad_M = np.zeros(N)
            k_vals = np.arange(N)
            j_vals = max_idx - k_vals
            valid = (j_vals >= 0) & (j_vals < N)
            grad_M[valid] = 2.0 * h[j_vals[valid]]
            
            grad_J = (M * grad_Q - Q * grad_M) / (M * M)
            h = h + eta * grad_J
            h = np.maximum(h, 0)
            s = np.sum(h)
            if s < 1e-12: break
            h = h / s
            
            curr_score = compute_score(h)
            if curr_score > best_score_local:
                best_score_local = curr_score
                best_h_local = h.copy()
                
        return best_h_local

    for _ in range(10):
        if time.time() > deadline:
            break
        h_rand = np.random.rand(N)
        h_rand = h_rand ** 2 # Sharpen
        h_opt = fast_grad_ascent(h_rand, steps=300)
        score_opt = compute_score(h_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt.copy()

    if best_h is None:
        return [1.0] * N
        
    return list(best_h)
