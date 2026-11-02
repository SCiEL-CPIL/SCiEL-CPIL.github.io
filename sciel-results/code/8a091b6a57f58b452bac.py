import numpy as np
import time
from scipy.optimize import minimize
import warnings

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def neg_score(h):
        s = score(h)
        if s < 1e-10:
            return 1e10
        return -s

    def optimize_h(initial_h, max_iter=200):
        # Normalize initial_h to sum 1
        h0 = np.maximum(initial_h, 0)
        s = np.sum(h0)
        if s == 0:
            h0 = np.ones(N) / N
        else:
            h0 = h0 / s
            
        # Use SLSQP to maximize score subject to sum(h)=1, h>=0
        # We minimize -score
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x} # x >= 0
        ]
        
        # Bounds
        bounds = [(0, 1)] * N
        
        # We use a soft-max surrogate for the max convolution to make it differentiable
        # However, scipy's minimize might struggle with non-diff max.
        # Let's try a direct approach first with Nelder-Mead which doesn't need gradients.
        
        best_h = h0.copy()
        best_s = score(h0)
        
        # Multiple local optimizations using Nelder-Mead
        # Nelder-Mead is robust for non-smooth functions
        
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the current best solution
            if restart == 0:
                h_start = h0.copy()
            else:
                h_start = best_h.copy()
                # Add noise
                noise = np.random.normal(0, 0.1, N)
                h_start = np.maximum(h_start + noise, 0)
                h_start = h_start / np.sum(h_start)
                
            try:
                res = minimize(
                    neg_score, 
                    h_start, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                h_opt = res.x
                h_opt = np.maximum(h_opt, 0)
                s_opt = np.sum(h_opt)
                if s_opt > 0:
                    h_opt = h_opt / s_opt
                    s_opt_val = score(h_opt)
                    if s_opt_val > best_s:
                        best_s = s_opt_val
                        best_h = h_opt.copy()
            except Exception as e:
                pass
        
        return best_h, best_s

    # Generate diverse initial points
    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Uniform
    h_uni = np.ones(N) / N
    h_opt, s_opt = optimize_h(h_uni)
    if s_opt > best_score:
        best_score = s_opt
        best_h = h_opt
        
    # 2. Sparse (Sidon-like)
    # Try to find a set with low auto-correlation
    # Greedy construction for low auto-correlation
    def greedy_low_corr(k):
        S = []
        for i in range(k):
            best_idx = -1
            best_max = np.inf
            for idx in range(N):
                if idx in S:
                    continue
                S_test = S + [idx]
                h_test = np.zeros(N)
                h_test[S_test] = 1.0
                conv = np.convolve(h_test, h_test, mode='full')
                # We want to minimize the max conv, but the peak at 0 is k.
                # We care about the off-diagonal peaks relative to the diagonal?
                # The score formula uses max_conv.
                # max_conv is always k (at shift 0).
                # So for binary vectors, score is 1/k.
                # This suggests binary vectors are bad if k > 1.
                # But the seed got 1.14.
                # This implies the seed found a non-binary vector.
                pass
            # This greedy is flawed for the current understanding.
            # Let's just use random sparse vectors.
            pass
        return S

    # 3. Random sparse vectors
    rng = np.random.RandomState(42)
    for i in range(20):
        if time.time() > deadline - 10.0:
            break
        k = rng.randint(2, 10)
        S = rng.choice(N, k, replace=False)
        h_init = np.zeros(N)
        h_init[S] = 1.0
        h_init = h_init / np.sum(h_init)
        
        h_opt, s_opt = optimize_h(h_init, max_iter=100)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
            
    # 4. Random dense vectors
    for i in range(20):
        if time.time() > deadline - 10.0:
            break
        h_init = rng.rand(N)
        h_init = h_init / np.sum(h_init)
        
        h_opt, s_opt = optimize_h(h_init, max_iter=100)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
            
    return list(best_h)
