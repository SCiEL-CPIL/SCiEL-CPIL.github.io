import time
import numpy as np
from scipy.optimize import linprog, minimize
from itertools import product, combinations_with_replacement
import sys

N = 50

# EVOLVE-BLOCK-START
def _objective(x, target_peak=1.0):
    """
    Objective for optimization.
    Maximize log(||x||_2^2) - log(||x||_1) - log(sup(||x*x||_infty / target_peak))
    Subject to: x >= 0, sum(x) = 1 (implicit in constraints or handled outside?)
    Let's use a smooth surrogate for the min.
    Actually, the score is J = ||x||_2^2 / (||x||_1 * ||x*x||_infty).
    We want to maximize J.
    
    If we fix ||x||_1 = 1, then we maximize ||x||_2^2 / ||x*x||_infty.
    
    Let's use a penalty method or direct optimization.
    Let f(x) = log(sum(x^2)) - log(sum(x)) - log(sum(exp(y_k))) where y_k = log(|(x*x)_k|) + log(target_peak)?
    No, let's just optimize the ratio directly with gradient descent or a black box optimizer on a transformed variable.
    
    Since the space is small (50 dims) and non-negative, we can use SLSQP or L-BFGS-B.
    However, the max operation in convolution sup is non-differentiable.
    We can approximate the sup with a soft-max:
    sup_k z_k ≈ exp((1/T) * sum_k log(exp(T * z_k)))
    For large T, this approaches the max.
    
    Let z_k = (x*x)_k.
    Objective: Minimize - [ log(sum(x^2)) - log(sum(x)) - log(SoftMax(z)) ]
    Or simply: Maximize J_smooth = sum(x^2) / (sum(x) * SoftMax(x*x))
    
    Let's implement a local search or a global optimization with restarts.
    Given the 150s budget, we can do:
    1. Try specific structured candidates (Sidon sets, spikes).
    2. Run a local optimizer from a few good starts.
    """
    x = np.asarray(x)
    if np.any(x < 0):
        x = np.abs(x) # Project back to non-negative
    
    norm_l1 = np.sum(x)
    if norm_l1 == 0:
        return 1e10 # Penalty
    
    norm_l2_sq = np.sum(x**2)
    
    conv = np.convolve(x, x, mode='full')
    
    # Smooth max of conv
    # T for soft max
    T = 50.0
    # Avoid log(0)
    conv_safe = np.maximum(conv, 1e-15)
    # log-sum-exp trick
    max_conv = np.max(conv_safe)
    log_sum_exp = max_conv + np.log(np.sum(np.exp(T * (conv_safe - max_conv)))) / T
    # Wait, soft max formula: (1/T) log sum exp(T z_i)
    # If T is large, it's max.
    # Let's compute soft_max_value = np.exp(log_sum_exp)
    
    # Actually, we want to maximize: norm_l2_sq / (norm_l1 * soft_max_value)
    # Taking log: log(norm_l2_sq) - log(norm_l1) - log(soft_max_value)
    # soft_max_value is positive.
    
    log_num = np.log(norm_l2_sq)
    log_denom = np.log(norm_l1) + log_sum_exp
    
    # We want to minimize negative log ratio
    return -(log_num - log_denom)

def _evaluate(h):
    """Calculate the true score for a candidate h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv == 0:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _generate_sidon_set(N):
    """
    Generate a Sidon set of size k in [0, N-1] or similar grid.
    A Sidon set has all pairwise sums distinct.
    This minimizes collisions in convolution.
    """
    # Greedy construction for Sidon set
    S = [0, 1, 4] # Initial
    # Try to add more
    for x in range(5, N):
        valid = True
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if x + S[i] == S[i] + S[j]: # x + S[i] == S[j] + S[i] -> x == S[j], already distinct
                    valid = False
                if x + S[j] == S[i] + S[j]: # x == S[i], distinct
                    valid = False
                # Check if new sums collide with existing sums
                # Existing sums are S[i]+S[j]
                # New sums are x+S[i], x+S[j]
                # We need x+S[i] != x+S[j] (true)
                # And x+S[i] != S[a]+S[b] for all a,b
                for a in range(len(S)):
                    for b in range(a, len(S)):
                        if x + S[i] == S[a] + S[b]:
                            valid = False
                            break
                        if x + S[j] == S[a] + S[b]:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
        if valid:
            S.append(x)
    return S

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    best_h = None
    best_score = 0.0
    
    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        if np.any(h < 0):
            h = np.abs(h)
        s = _evaluate(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # 1. Simple candidates
    # Spike
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    try_candidate(h_spike)
    
    # Uniform
    h_uniform = np.ones(N)
    try_candidate(h_uniform)
    
    # Sidon set based
    S = _generate_sidon_set(N)
    h_sidon = np.zeros(N)
    for idx in S:
        h_sidon[idx] = 1.0
    try_candidate(h_sidon)
    
    # Try random sparse vectors
    np.random.seed(42)
    for _ in range(100):
        if time.time() > deadline: break
        k = np.random.randint(3, 15)
        indices = np.random.choice(N, k, replace=False)
        h_rand = np.zeros(N)
        h_rand[indices] = 1.0
        try_candidate(h_rand)
        
    # 2. Local Optimization from best candidates
    # We will optimize in the space of log-heights or directly with bounds.
    # Using scipy.optimize.minimize with method='L-BFGS-B' or 'SLSQP'
    
    def run_local_opt(h_start):
        x0 = np.log(np.maximum(h_start, 1e-6))
        
        def obj(log_x):
            x = np.exp(log_x)
            # Normalize L1 to 1? 
            # The objective is scale-invariant for L2^2/L1/SupConv?
            # J(c x) = c^2 ||x||^2 / (c ||x||_1 * c ||x*x||_inf) = J(x).
            # So we can fix ||x||_1 = 1.
            # However, minimizing in log space without constraint might drift.
            # Let's add a penalty for L1 != 1 or just normalize inside.
            x_norm = x / np.sum(x)
            return _objective(x_norm)
        
        # We want to minimize negative log ratio, so _objective returns negative log ratio.
        # Wait, _objective returns -log(J_smooth). So minimizing it maximizes J_smooth.
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-4})
            x_opt = np.exp(res.x)
            x_opt = x_opt / np.sum(x_opt) # Ensure L1=1
            try_candidate(x_opt)
        except Exception:
            pass

    # Run optimization from the best candidate so far
    if best_h is not None:
        run_local_opt(best_h)
        
    # Run optimization from Sidon
    run_local_opt(h_sidon)
    
    # Run optimization from a few random sparse starts
    for _ in range(5):
        if time.time() > deadline: break
        k = np.random.randint(5, 10)
        indices = np.random.choice(N, k, replace=False)
        h_rand = np.zeros(N)
        h_rand[indices] = 1.0
        run_local_opt(h_rand)
        
    # 3. High precision polish
    # If we found a good candidate, try to refine it with higher T in soft max or exact local solve
    if best_h is not None:
        # Try a finer local search with SLSQP on the normalized vector
        x0 = best_h / np.sum(best_h)
        
        def obj_exact(x):
            x = np.abs(x)
            l1 = np.sum(x)
            if l1 == 0: return 1e10
            l2_sq = np.sum(x**2)
            conv = np.convolve(x, x, mode='full')
            sup = np.max(conv)
            if sup == 0: return 1e10
            # Use soft max with high T for gradient
            T = 100.0
            conv_safe = np.maximum(conv, 1e-15)
            max_conv = np.max(conv_safe)
            log_sum_exp = max_conv + np.log(np.sum(np.exp(T * (conv_safe - max_conv)))) / T
            log_j = np.log(l2_sq) - np.log(l1) - log_sum_exp
            return -log_j
        
        try:
            res = minimize(obj_exact, x0, method='SLSQP', bounds=[(0, None)]*N, options={'maxiter': 200, 'ftol': 1e-6})
            x_opt = res.x
            x_opt = x_opt / np.sum(x_opt)
            try_candidate(x_opt)
        except Exception:
            pass

    # Final check
    if best_h is None:
        return [1.0] + [0.0]*(N-1)
    
    # Return the best heights found
    # The problem asks for "sequence of nonnegative step heights"
    # Score is lower bound on C2? No, "score is the raw value (higher is better). The search score is value/seed".
    # So we just return the array.
    return best_h.tolist()

# EVOLVE-BLOCK-END
