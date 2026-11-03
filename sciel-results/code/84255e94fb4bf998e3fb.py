import numpy as np
import time
import sys
from scipy.optimize import minimize, LinearConstraint

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return deadline - elapsed()
    
    # Score function with L1=1 normalization (scale-invariant form)
    # For L1=1: score = L2_sq / (DELTA * max_conv)
    # We maximize this, which is equivalent to maximizing L2_sq / max_conv
    
    def score_normalized(h):
        """Score assuming h is normalized to L1=1."""
        h = np.asarray(h, dtype=np.float64)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0.0
        return l2_sq / (DELTA * max_conv)
    
    # Smooth surrogate: soft-max of convolution using log-sum-exp
    def soft_max_conv(h, beta=100.0):
        """Compute a smooth approximation to max(conv(h,h)) using log-sum-exp."""
        h_conv = np.convolve(h, h, mode='full')
        # Log-sum-exp for stability
        m = np.max(h_conv)
        if m <= 0:
            return 0.0
        # soft-max = m + (1/beta) * log(sum(exp(beta*(x-m))))
        # But for large beta, this approaches max
        # Use a simpler approach: weighted max with temperature
        # Actually, let's just use the hard max for now but make it differentiable
        # by tracking which index achieves max and using that in gradient
        return m
    
    # Parametrization: use simplex via softmax
    # h_i = exp(x_i) / sum(exp(x_j)), so sum(h) = 1 automatically
    # This removes the L1=1 constraint and h >= 0 constraint
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def neg_score_log(x):
        """Objective in log-space: minimize -score(softmax(x))"""
        h = softmax(x)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0.0
        return -(l2_sq / (DELTA * max_conv))
    
    # Generate diverse initializations
    x_idx = np.arange(N)
    best_h = None
    best_score = -np.inf
    
    def update_best(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=np.float64)
        if np.sum(h) <= 0:
            return
        h_norm = h / np.sum(h)
        s = score_normalized(h_norm)
        if s > best_score:
            best_score = s
            best_h = h_norm.copy()
    
    # 1. Uniform
    h_unif = np.ones(N) / N
    update_best(h_unif)
    
    # 2. Gaussian peaks
    for c in np.linspace(2, N-2, 8):
        for w in np.linspace(1, N/3, 8):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            update_best(h_g)
    
    # 3. Triangular
    for c in np.linspace(2, N-2, 8):
        for w in np.linspace(1, N/2, 8):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                update_best(h_t)
    
    # 4. Sparse peaks
    for _ in range(50):
        if elapsed() > 10:
            break
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        # Add small noise to break symmetry
        h_s += np.random.rand(N) * 0.05
        update_best(h_s)
    
    # 5. Random smooth
    for _ in range(50):
        if elapsed() > 15:
            break
        h_r = np.random.rand(N)
        kernel = np.ones(7) / 7
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 0:
            update_best(h_r)
    
    # 6. Two-peak structures (Sidon-like: minimize autoconv overlap)
    for _ in range(30):
        if elapsed() > 20:
            break
        h_2p = np.zeros(N)
        c1 = np.random.randint(5, N//2)
        c2 = np.random.randint(N//2 + 1, N - 5)
        w1 = np.random.uniform(1, N/4)
        w2 = np.random.uniform(1, N/4)
        h_2p += np.exp(-0.5 * ((x_idx - c1) / w1)**2)
        h_2p += np.exp(-0.5 * ((x_idx - c2) / w2)**2)
        update_best(h_2p)
    
    # Optimization using SLSQP on log-space (softmax parametrization)
    # No explicit constraints needed since softmax ensures h >= 0 and sum(h) = 1
    
    # Start from best_h
    h_start = best_h.copy()
    # Convert to log-space: x_i = log(h_i) + constant (doesn't matter for softmax)
    h_safe = np.maximum(h_start, 1e-15)
    x0 = np.log(h_safe)
    
    # Multiple restarts with different initializations
    inits = [x0]
    
    # Add log-space versions of other good initializations
    for c in np.linspace(2, N-2, 5):
        for w in np.linspace(1, N/3, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g = np.maximum(h_g, 1e-15)
            inits.append(np.log(h_g))
    
    for _ in range(20):
        if elapsed() > 25:
            break
        h_r = np.random.rand(N)
        h_r = np.maximum(h_r, 1e-15)
        inits.append(np.log(h_r))
    
    # Optimize from each init
    for x_init in inits:
        if elapsed() > deadline - 20:
            break
        
        try:
            res = minimize(neg_score_log, x_init, method='BFGS', 
                          options={'maxiter': 500, 'gtol': 1e-8})
            if res.success:
                h_candidate = softmax(res.x)
                s_candidate = score_normalized(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
    
    # Polish with tighter tolerance
    if elapsed() < deadline - 15:
        h_start = best_h.copy()
        h_safe = np.maximum(h_start, 1e-15)
        x0 = np.log(h_safe)
        
        try:
            res = minimize(neg_score_log, x0, method='BFGS',
                          options={'maxiter': 2000, 'gtol': 1e-12})
            if res.success:
                h_candidate = softmax(res.x)
                s_candidate = score_normalized(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
    
    # Final local refinement: coordinate descent on the simplex
    # For each coordinate, optimize while keeping others fixed
    if elapsed() < deadline - 10:
        h_refine = best_h.copy()
        for iteration in range(50):
            if elapsed() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if elapsed() > deadline - 5:
                    break
                # Optimize h_i while keeping sum(h) = 1
                # h_i = t, others fixed, sum(others) = S, so t = 1 - S
                # Actually, we need to re-normalize
                # Let's do: fix all except i, optimize the ratio
                # L2_sq = h_i^2 + sum_{j!=i} h_j^2
                # Conv depends on h_i nonlinearly
                # This is complex, so skip for now
                pass
            if not improved:
                break
    
    # Ensure best_h is valid and normalized
    if best_h is None:
        best_h = np.ones(N) / N
    else:
        best_h = np.maximum(best_h, 0)
        s = np.sum(best_h)
        if s > 0:
            best_h = best_h / s
    
    return list(best_h)
