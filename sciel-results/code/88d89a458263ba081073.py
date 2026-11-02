"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _compute_overlaps(h):
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        overlaps[k] = np.dot(h, 1.0 - np.roll(h, -k))
    return overlaps

def _max_overlap(h):
    return np.max(_compute_overlaps(h))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # The overlap at shift k is:
    # overlap_k = sum_i h[i] * (1 - h[(i-k) % N])
    #           = sum_i h[i] - sum_i h[i] * h[(i-k) % N]
    #           = N/2 - autocorr_k
    # where autocorr_k = sum_i h[i] * h[(i-k) % N]
    
    # We want to minimize max_k overlap_k = N/2 - min_k autocorr_k
    # So we want to maximize min_k autocorr_k.
    
    # Equivalently, minimize t subject to:
    # N/2 - autocorr_k <= t  for all k
    # autocorr_k >= N/2 - t  for all k
    # sum_i h_i = N/2
    # 0 <= h_i <= 1
    
    # autocorr_k = sum_i h[i] * h[(i-k) % N]
    # This is a quadratic term, so it's not directly LP.
    
    # However, we can use the fact that for the purpose of minimizing max overlap,
    # we can try a different approach. Let's think about what structure minimizes max overlap.
    
    # For k=0: overlap_0 = sum_i h[i]*(1-h[i]) = N/2 - sum_i h[i]^2
    # To minimize overlap_0, we want to maximize sum_i h[i]^2, which means making h as "binary" as possible.
    
    # For other k, overlap_k = N/2 - autocorr_k. To minimize this, we want autocorr_k to be large.
    # autocorr_k = sum_i h[i]*h[i-k]. If h is periodic with period p, then autocorr_k is large when k is a multiple of p.
    
    # A good candidate: h is a block of 1s followed by a block of 0s.
    # If h[i] = 1 for i < m and 0 otherwise, with m = N/2 = 50.
    # Then overlap_k = number of i where h[i]=1 and h[i-k]=0.
    # For k=0: overlap_0 = 0 (since h[i]*(1-h[i])=0)
    # For k=1: overlap_1 = number of i where h[i]=1 and h[i-1]=0 = 1 (only i=m)
    # For k=2: overlap_2 = 2
    # ...
    # For k=m: overlap_m = m = 50
    # So max overlap = 50. That's terrible.
    
    # Better: alternate 1s and 0s. h[i] = 1 if i even, 0 if i odd.
    # sum h = 50. 
    # overlap_0 = 0
    # overlap_1: h[i]*(1-h[i-1]). For i even, h[i]=1, h[i-1]=0, so 1*1=1. For i odd, h[i]=0, so 0. Sum = 50.
    # overlap_2: h[i]*(1-h[i-2]). h[i-2] has same parity as h[i], so 1-h[i-2] = 1-h[i]. For i even: 1*(1-1)=0. For i odd: 0. Sum=0.
    # So max overlap = 50. Still bad.
    
    # Let's try h[i] = 0.5 for all i.
    # overlap_k = sum_i 0.5 * 0.5 = 25 for all k. Max = 25.
    
    # Can we do better than 25?
    # We need max_k (N/2 - autocorr_k) < 25, i.e., autocorr_k > 25 for all k.
    # autocorr_0 = sum h[i]^2 >= (sum h[i])^2 / N = 2500/100 = 25 by Cauchy-Schwarz.
    # So autocorr_0 >= 25, meaning overlap_0 <= 25. Equality when all h[i] = 0.5.
    # For k != 0, we need autocorr_k > 25.
    # But autocorr_k = sum h[i] h[i-k]. If all h[i] = 0.5, autocorr_k = 25 for all k.
    # To get autocorr_k > 25 for some k, we need variation. But then autocorr_0 > 25, so overlap_0 < 25.
    # The question is whether we can keep all autocorr_k >= 25 (i.e., all overlap_k <= 25) while having variation.
    
    # Actually, let's reconsider. We want to minimize max_k overlap_k.
    # Let's use the LP approach with a different formulation.
    
    # Let x_i = h[i] - 0.5. Then sum x_i = 0, -0.5 <= x_i <= 0.5.
    # overlap_k = sum (0.5+x_i)(1 - (0.5 + x_{i-k}))
    #           = sum (0.5+x_i)(0.5 - x_{i-k})
    #           = sum (0.25 - 0.5 x_{i-k} + 0.5 x_i - x_i x_{i-k})
    #           = N/4 - 0.5 sum x_{i-k} + 0.5 sum x_i - sum x_i x_{i-k}
    #           = 25 - sum x_i x_{i-k}  (since sum x_i = 0)
    
    # So overlap_k = 25 - sum_i x_i x_{i-k}.
    # We want to minimize max_k (25 - sum_i x_i x_{i-k}) = 25 - min_k sum_i x_i x_{i-k}.
    # So we want to maximize min_k autocorr_k(x) where autocorr_k(x) = sum_i x_i x_{i-k}.
    
    # This is still quadratic. But for small N, we can try to use scipy's optimize with the original formulation.
    
    # Let's try a direct numerical optimization approach using scipy.
    from scipy.optimize import minimize
    
    def objective_and_grad(vars_vec):
        x = vars_vec[:N]
        h = 0.5 + x
        overlaps = np.zeros(N)
        for k in range(N):
            overlaps[k] = np.dot(h, 1.0 - np.roll(h, -k))
        m = np.max(overlaps)
        # Soft max approximation for gradient
        beta = 100.0
        exp_overlaps = np.exp((overlaps - m) * beta)
        sum_exp = np.sum(exp_overlaps)
        weights = exp_overlaps / sum_exp
        
        grad = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-10:
                g_k = (1.0 - np.roll(h, -k)) - np.roll(h, k)
                grad += weights[k] * g_k
        
        # Constraint gradient: sum h = N/2, so sum x = 0. 
        # We'll handle constraints via projection.
        return m, grad
    
    def project(x):
        h = 0.5 + x
        # Project h to [0,1]^N with sum = N/2
        # Use bisection on offset
        lo = -1.0
        hi = 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            h_proj = np.clip(h - mid, 0, 1)
            s = np.sum(h_proj)
            if s > N / 2:
                lo = mid
            else:
                hi = mid
        alpha = (lo + hi) / 2
        h_proj = np.clip(h - alpha, 0, 1)
        # Fine tune
        for _ in range(10):
            s = np.sum(h_proj)
            diff = N / 2 - s
            if abs(diff) < 1e-12:
                break
            idx = np.where((h_proj > 1e-9) & (h_proj < 1 - 1e-9))[0]
            if len(idx) == 0:
                break
            h_proj[idx] += diff / len(idx)
            h_proj = np.clip(h_proj, 0, 1)
        return h_proj
    
    # Start from uniform
    h0 = np.full(N, 0.5)
    x0 = np.zeros(N)
    
    best_h = h0.copy()
    best_val = _max_overlap(best_h)
    
    # Try multiple random starts
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if s == 0:
            x_start = np.zeros(N)
        else:
            x_start = np.random.uniform(-0.3, 0.3, N)
            x_start -= np.mean(x_start)
        
        h_start = 0.5 + x_start
        h_start = project(h_start)
        
        result = minimize(
            lambda x: objective_and_grad(x)[0],
            x_start - 0.5,  # Wait, objective takes x where h=0.5+x, so pass x
            method='L-BFGS-B',
            jac=lambda x: objective_and_grad(x)[1],
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        x_opt = result.x
        h_opt = 0.5 + x_opt
        h_opt = project(h_opt)
        val = _max_overlap(h_opt)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Local refinement
    if time.time() < deadline - 10:
        h_curr = best_h.copy()
        x_curr = h_curr - 0.5
        beta = 50.0
        lr = 0.05
        for i in range(500):
            if time.time() > deadline - 5:
                break
            
            m, grad = objective_and_grad(x_curr)
            # Gradient step in x-space
            x_new = x_curr - lr * grad
            h_new = 0.5 + x_new
            h_new = project(h_new)
            val_new = _max_overlap(h_new)
            
            if val_new < best_val - 1e-12:
                best_val = val_new
                best_h = h_new.copy()
                x_curr = h_curr - 0.5
                beta *= 1.01
                lr *= 0.995
            else:
                lr *= 0.5
                beta *= 1.1
                if lr < 1e-8:
                    break
    
    return best_h
# EVOLVE-BLOCK-END
