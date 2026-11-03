import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _generate_sidon_supports(N, num_candidates=50):
    """
    Generate several candidate supports that are 'Sidon-like' (minimize sum collisions).
    """
    supports = []
    
    # Greedy Sidon construction
    def greedy_sidon(start_offset=0, max_size=None):
        S = []
        sums = set()
        for i in range(start_offset, N):
            # Check if adding i creates a collision
            # New sums: i + s for s in S, and i + i
            new_sums = [i + s for s in S] + [2*i]
            if not any(ns in sums for ns in new_sums):
                S.append(i)
                sums.update(new_sums)
                if max_size and len(S) >= max_size:
                    break
        return S

    # 1. Standard greedy
    supports.append(greedy_sidon(0))
    
    # 2. Greedy with different starts (shifted)
    for start in range(1, 5):
        supports.append(greedy_sidon(start))
        
    # 3. Randomized Greedy
    np.random.seed(123)
    for _ in range(num_candidates):
        S = []
        sums = set()
        candidates = list(range(N))
        np.random.shuffle(candidates)
        for i in candidates:
            new_sums = [i + s for s in S] + [2*i]
            if not any(ns in sums for ns in new_sums):
                S.append(i)
                sums.update(new_sums)
                if len(S) >= 15: # Cap size for speed, though max is ~7
                    break
        if S:
            supports.append(S)
            
    # 4. Structured sparse (pairs)
    for k in range(3, 10):
        for offset in range(1, 8):
            h_pair = np.zeros(N)
            center = N // 2
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    h_pair[pos1] = 1.0
                if 0 <= pos2 < N:
                    h_pair[pos2] = 1.0
            if np.sum(h_pair) > 0:
                supports.append(np.nonzero(h_pair)[0].tolist())

    # Remove duplicates
    unique_supports = []
    seen = set()
    for s in supports:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_supports.append(s)
            
    return unique_supports

def _optimize_weights_for_support(support, h_init=None, beta=50.0, maxiter=200):
    """
    Optimize weights on a fixed support to maximize the ratio.
    Variables: w (length k)
    Objective: Minimize -log(R) = log(sum(w)*max_conv) - log(sum(w^2))
    With sum(w)=1, minimize log(max_conv) - log(sum(w^2))
    """
    k = len(support)
    if k == 0:
        return None
    
    # Map support indices to a smaller vector
    # h is length N, but we only optimize w of length k
    
    if h_init is None:
        w0 = np.ones(k) / k
    else:
        w0 = np.array([h_init[i] for i in support])
        if np.sum(w0) < 1e-10:
            w0 = np.ones(k) / k
        else:
            w0 = w0 / np.sum(w0)

    def objective(w):
        # Ensure non-negative and sum to 1 (approx, L-BFGS-B handles bounds)
        w = np.clip(w, 0, None)
        if np.sum(w) < 1e-10:
            return 1e10
        w_norm = w / np.sum(w)
        
        # Construct full h
        h = np.zeros(N)
        h[support] = w_norm
        
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-10:
            return 1e10
        
        h_conv = np.convolve(h, h, mode='full')
        
        # Smooth max using log-sum-exp
        # max(x) ~ (1/beta) * log(sum(exp(beta * x)))
        # We want to minimize log(max_conv) - log(l2_sq)
        # log(max_conv) ~ log( (1/beta) * log(sum(exp(beta * h_conv))) )
        # Actually, just use the smooth approximation directly in the objective
        
        # To avoid overflow in exp, subtract max
        max_conv = np.max(h_conv)
        if max_conv < 1e-10:
            return 1e10
            
        # Stable log-sum-exp
        beta_safe = beta
        shifted = h_conv - max_conv
        log_sum_exp = max_conv + np.log(np.sum(np.exp(beta_safe * shifted)) / beta_safe)
        
        # Objective: log(max_conv) - log(l2_sq)
        # Using smooth approx for max_conv
        return log_sum_exp - np.log(l2_sq)

    def gradient(w):
        # Numerical gradient is safer for this complex objective
        eps = 1e-7
        grad = np.zeros(k)
        f0 = objective(w)
        for i in range(k):
            w_plus = w.copy()
            w_plus[i] += eps
            f_plus = objective(w_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    bounds = [(0.0, 1.0)] * k
    
    try:
        res = opt.minimize(
            objective, 
            w0, 
            method='L-BFGS-B', 
            bounds=bounds,
            jac=gradient,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        
        if res.success:
            w_opt = res.x
            w_opt = np.clip(w_opt, 0, None)
            if np.sum(w_opt) < 1e-10:
                return None
            w_opt = w_opt / np.sum(w_opt)
            
            h_opt = np.zeros(N)
            h_opt[support] = w_opt
            return h_opt
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_h[0] = 1.0 # Delta function gives score 1
    best_score = _score_raw(best_h)
    
    # Generate candidate supports
    supports = _generate_sidon_supports(N, num_candidates=30)
    
    # Sort supports by size (larger Sidon sets generally better)
    supports.sort(key=lambda x: len(x), reverse=True)
    
    # Evaluate and refine each support
    for support in supports:
        if time.time() > deadline - 20:
            break
            
        # Initial uniform weights
        h_init = np.zeros(N)
        h_init[support] = 1.0 / len(support)
        
        # Quick check with uniform weights
        s_uniform = _score_raw(h_init)
        if s_uniform > best_score:
            best_score = s_uniform
            best_h = h_init.copy()
            
        # Refine weights
        h_opt = _optimize_weights_for_support(support, h_init, beta=50.0, maxiter=100)
        
        if h_opt is not None:
            s_opt = _score_raw(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
    # Final polish: Local search on the best solution found
    if time.time() < deadline - 10:
        # Try a few random perturbations of the best solution
        for _ in range(5):
            h_perturbed = best_h.copy()
            # Add small noise to non-zero elements
            mask = h_perturbed > 1e-6
            h_perturbed[mask] += np.random.normal(0, 0.01, np.sum(mask))
            h_perturbed = np.clip(h_perturbed, 0, None)
            if np.sum(h_perturbed) > 0:
                h_perturbed /= np.sum(h_perturbed)
                s_pert = _score_raw(h_perturbed)
                if s_pert > best_score:
                    best_score = s_pert
                    best_h = h_perturbed.copy()
                    
    return list(best_h)

# EVOLVE-BLOCK-END
