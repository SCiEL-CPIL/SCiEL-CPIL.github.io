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

def compute_ratio_and_grad(h, support):
    """Compute ratio and its gradient w.r.t. h[support]."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0, np.zeros(len(support))
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0, np.zeros(len(support))
    
    ratio = h2_sq / conv_inf
    
    # Gradient of ratio w.r.t. h
    # ratio = ||h||_2^2 / ||h*h||_inf (with h normalized so sum=1)
    # Let's compute numerically for reliability
    grad = np.zeros(N)
    eps = 1e-7
    for idx in range(len(support)):
        h_plus = h.copy()
        h_plus[support[idx]] += eps
        r_plus = compute_ratio(h_plus)
        h_minus = h.copy()
        h_minus[support[idx]] -= eps
        r_minus = compute_ratio(h_minus)
        grad[support[idx]] = (r_plus - r_minus) / (2 * eps)
    
    return ratio, grad[support]

def local_search_polish(h, support, max_iter=200, tol=1e-10):
    """Local search: try small perturbations to improve the ratio."""
    h = h.copy()
    k = len(support)
    if k < 2:
        return h
    
    best_score = compute_ratio(h)
    
    for it in range(max_iter):
        improved = False
        # Try swapping weight between two indices with adaptive step size
        for i in range(k):
            for j in range(i+1, k):
                # Try moving a small amount from i to j
                for delta in [0.005, 0.002, 0.001, -0.005, -0.002, -0.001]:
                    h_new = h.copy()
                    hi = h[support[i]]
                    hj = h[support[j]]
                    new_hi = hi + delta
                    new_hj = hj - delta
                    if new_hi >= 0 and new_hj >= 0:
                        h_new[support[i]] = new_hi
                        h_new[support[j]] = new_hj
                        s = np.sum(h_new)
                        if s > 1e-15:
                            h_new = h_new / s
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                h = h_new
                                best_score = sc
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try single index adjustments
            for i in range(k):
                for delta in [0.005, 0.002, 0.001, -0.005, -0.002, -0.001]:
                    h_new = h.copy()
                    hi = h[support[i]]
                    new_hi = hi + delta
                    if new_hi >= 0:
                        h_new[support[i]] = new_hi
                        s = np.sum(h_new)
                        if s > 1e-15:
                            h_new = h_new / s
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                h = h_new
                                best_score = sc
                                improved = True
                                break
                if improved:
                    break
            
            if not improved:
                break
    
    return h

def gradient_polish(h, support, max_iter=500, tol=1e-12):
    """Polish using gradient-based optimization on the ratio objective."""
    h = h.copy()
    k = len(support)
    if k < 2:
        return h
    
    # Use L-BFGS-B to maximize ratio directly
    def neg_ratio_and_grad(x):
        h_full = np.zeros(N)
        h_full[support] = np.maximum(x, 0)
        s = np.sum(h_full)
        if s < 1e-15:
            return 1e10, np.zeros(k)
        h_norm = h_full / s
        r, _ = compute_ratio_and_grad(h_norm, support)
        return -r, np.zeros(k)  # Placeholder gradient
    
    # Better: use the analytic gradient
    def neg_ratio(x):
        h_full = np.zeros(N)
        h_full[support] = np.maximum(x, 0)
        s = np.sum(h_full)
        if s < 1e-15:
            return 1e10
        h_norm = h_full / s
        r, _ = compute_ratio_and_grad(h_norm, support)
        return -r
    
    x0 = h[support].copy()
    x0 = np.maximum(x0, 0)
    s0 = np.sum(x0)
    if s0 > 1e-15:
        x0 = x0 / s0
    
    try:
        res = minimize(neg_ratio, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        x_opt = np.maximum(res.x, 0)
        s_opt = np.sum(x_opt)
        if s_opt > 1e-15:
            x_opt = x_opt / s_opt
            h_new = np.zeros(N)
            h_new[support] = x_opt
            sc = compute_ratio(h_new)
            if sc > compute_ratio(h):
                h = h_new
    except:
        pass
    
    return h

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

def optimize_weights_for_support(support, beta=100.0):
    """
    Optimize weights h_i for i in support to maximize the ratio.
    Uses smooth max surrogate with proper gradient handling on the simplex.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform weights
    x0 = np.ones(k) / k
    
    def objective_and_grad(x):
        """Compute smooth max of convolution and its gradient w.r.t x (on simplex)."""
        h = np.zeros(N)
        h[support] = np.maximum(x, 0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(k)
        h = h / s
        conv = fftconvolve(h, h, mode='full')
        
        # Smooth max using log-sum-exp
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        log_sum_exp = max_c + np.log(sum_exp)
        obj = log_sum_exp / beta
        
        # Compute gradient properly
        # d obj / d x_i where h = x/s
        # Let's use automatic differentiation via finite differences for reliability
        grad = np.zeros(k)
        eps = 1e-7
        for idx in range(k):
            x_plus = x.copy()
            x_plus[idx] += eps
            h_plus = np.zeros(N)
            h_plus[support] = np.maximum(x_plus, 0)
            s_plus = np.sum(h_plus)
            if s_plus > 1e-15:
                h_plus = h_plus / s_plus
                conv_plus = fftconvolve(h_plus, h_plus, mode='full')
                max_plus = np.max(conv_plus)
                if max_plus > 1e-15:
                    shifted_plus = conv_plus - max_plus
                    exp_plus = np.exp(beta * shifted_plus)
                    sum_plus = np.sum(exp_plus)
                    obj_plus = (max_plus + np.log(sum_plus)) / beta
                    grad[idx] = (obj_plus - obj) / eps
                else:
                    grad[idx] = 0
            else:
                grad[idx] = 0
        
        return obj, grad
    
    # Use L-BFGS-B with simplex projection
    # We'll parameterize as x = softmax(z) to ensure x >= 0 and sum(x) = 1
    def softmax(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    
    def objective_z(z):
        x = softmax(z)
        obj, _ = objective_and_grad(x)
        return obj
    
    def grad_z(z):
        x = softmax(z)
        obj, grad_x = objective_and_grad(x)
        # d softmax(z)_i / dz_j = x_i * (delta_ij - x_j)
        # So grad_z_j = sum_i grad_x_i * x_i * (delta_ij - x_j)
        # = x_j * grad_x_j - x_j * sum_i grad_x_i * x_i
        sum_grad_x_x = np.sum(grad_x * x)
        grad_z = x * (grad_x - sum_grad_x_x)
        return grad_z
    
    # Optimize in z-space (unconstrained)
    z0 = np.zeros(k)  # softmax(0) = uniform
    try:
        res = minimize(objective_z, z0, jac=grad_z, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        z_opt = res.x
        x_opt = softmax(z_opt)
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        
        # Local search polish
        h_opt = local_search_polish(h_opt, support, max_iter=100)
        
        sc = compute_ratio(h_opt)
        return h_opt, sc
    except Exception as e:
        # Fallback to uniform
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        h_opt = local_search_polish(h_opt, support, max_iter=50)
        sc = compute_ratio(h_opt)
        return h_opt, sc

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending - larger sets have more room for optimization
    sidon_sets.sort(key=lambda s: len(s), reverse=True)
    
    # Try optimizing weights for each Sidon set
    for s in sidon_sets:
        if time.time() > deadline - 20.0:
            break
        
        # Try multiple beta values
        for beta in [30.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 20.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
        
        # Also try with a few random initializations
        if time.time() > deadline - 20.0:
            break
        
        np.random.seed(hash(tuple(s)) % 2**32)
        for _ in range(3):
            if time.time() > deadline - 20.0:
                break
            k = len(s)
            x_init = np.random.dirichlet(np.ones(k))
            h_init = np.zeros(N)
            h_init[s] = x_init
            h_opt, sc = optimize_weights_for_support(s, beta=100.0)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()

    # Final polish on the best solution
    if time.time() < deadline - 10.0:
        # Find support of best_h
        support = [i for i in range(N) if best_h[i] > 1e-10]
        if len(support) >= 2:
            # Gradient polish first
            best_h = gradient_polish(best_h, support, max_iter=300, tol=1e-12)
            sc = compute_ratio(best_h)
            if sc > best_score:
                best_score = sc
            
            # Then local search polish
            best_h = local_search_polish(best_h, support, max_iter=300, tol=1e-12)
            best_score = compute_ratio(best_h)

    return list(best_h)
