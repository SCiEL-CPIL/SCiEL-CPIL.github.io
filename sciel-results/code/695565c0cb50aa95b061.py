import numpy as np
import time
from scipy.optimize import minimize, L_BFGS_B
import sys

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the raw score: L2^2 / (L1 * Sup(AutoConv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    # Discrete convolution h * h
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _score_soft(h, beta):
    """
    Compute a smoothed version of the score using log-sum-exp for the max operation.
    Score = L2^2 / (L1 * SoftMax(Conv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    
    # Soft max of h_conv
    # To avoid overflow, subtract max first
    max_conv = np.max(h_conv)
    if max_conv == 0:
        return 0.0
    # log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    # We want SoftMax = (1/beta) * log(sum(exp(beta * h_conv)))
    # This is equal to max(h_conv) + (1/beta) * log(sum(exp(beta * (h_conv - max(h_conv)))))
    shifted = beta * (h_conv - max_conv)
    # Clip to avoid underflow issues in exp, though exp(-large) is 0 which is fine
    # np.logsumexp is more stable
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_max_val = max_conv + (1.0 / beta) * log_sum_exp
    
    # The actual autoconv peak in the continuous sense is DELTA * max_conv
    # So the denominator is L1 * DELTA * soft_max_val
    autoconv_max = DELTA * soft_max_val
    if autoconv_max == 0:
        return 0.0
    
    return (l2_sq / (l1 * autoconv_max))

def _generate_sidon_set(n, k):
    """
    Greedy generation of a Sidon-like set.
    """
    S = [0]
    sums = set()
    for i in range(1, n):
        if len(S) >= k:
            break
        valid = True
        for s in S:
            val = s + i
            if val in sums:
                valid = False
                break
        if valid:
            for s in S:
                sums.add(s + i)
            S.append(i)
    return S

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Parametric search for good initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Gaussians
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            inits.append(h_gauss)
            
    # Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                inits.append(h_tri)

    # 2. Sparse Sidon-like initializations
    # Try different sizes
    for k in range(5, 20):
        # Generate multiple random Sidon sets
        for _ in range(5):
            if time.time() > deadline: break
            sidon_idx = _generate_sidon_set(N, k)
            if len(sidon_idx) >= 2:
                h_sidon = np.zeros(N)
                h_sidon[sidon_idx] = 1.0
                # Add small noise to break symmetry
                h_sidon += np.random.rand(N) * 0.01
                inits.append(h_sidon)
                
    # 3. Random sparse initialization
    for _ in range(50):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        inits.append(h_rand)

    # 4. Random smooth
    for _ in range(20):
        if time.time() > deadline: break
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        inits.append(h_r)

    # Optimization Loop
    bounds = [(1e-8, 10.0)] * N  # Strictly positive to avoid log issues if we switched, but we use h directly. 
    # Actually, for L-BFGS-B, we can use (0, inf) but numerical stability is better with small lower bound.
    # However, the objective is scale invariant, so we should normalize or fix scale.
    # Let's fix scale by normalizing inside the objective function? No, that complicates gradients.
    # Better: Add constraint sum(h)=1? L-BFGS-B doesn't support equality constraints.
    # Alternative: Optimize in log-space?
    # Let's try optimizing h directly with L-BFGS-B and just rely on the fact that the ratio is scale invariant?
    # If I scale h, the score changes as 1/c. So the optimizer will try to make h small.
    # We MUST fix the scale.
    # We can fix h[0] = 1? No, h[0] could be 0.
    # We can fix sum(h) = 1 by using a Lagrangian or simply normalizing after each step?
    # Simplest robust method for L-BFGS-B: 
    # Use variables x where h_i = exp(x_i). 
    # Objective: -Score(exp(x)).
    # But Score(exp(x)) is scale invariant? No.
    # Score(c * exp(x)) = Score(exp(x)) / c.
    # If we use x, scaling is just adding a constant to all x.
    # The gradient with respect to x will push towards -infinity (smaller h) to increase score.
    # So we must constrain the scale in x-space.
    # Constraint: sum(exp(x)) = 1.
    # This is hard for L-BFGS-B.
    
    # Let's go back to SLSQP which handles constraints, but use the SOFT objective to make it smoother.
    # SLSQP is slower but handles sum(h)=1 and h>=0 well.
    # We will use SLSQP with the soft-max objective.
    
    def neg_obj_soft(h, beta):
        return -_score_soft(h, beta)
        
    def neg_obj_raw(h):
        return -_score_raw(h)

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h} # h >= 0
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Annealing betas
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for h_init in inits:
        if time.time() > deadline: break
        
        # Normalize init
        h_init = np.asarray(h_init, dtype=np.float64)
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        # Ensure bounds
        h_init = np.clip(h_init, 0, 1)
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        current_h = h_init
        
        # Multi-stage optimization with softening
        for beta in betas:
            if time.time() > deadline: break
            
            try:
                res = minimize(
                    lambda h: neg_obj_soft(h, beta), 
                    current_h, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints, 
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res.success:
                    current_h = res.x
                    current_h = np.maximum(current_h, 0)
                    if np.sum(current_h) > 0:
                        current_h = current_h / np.sum(current_h)
            except Exception:
                pass
        
        # Final polish with exact (non-smooth) objective
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    neg_obj_raw, 
                    current_h, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints, 
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res.success:
                    current_h = res.x
                    current_h = np.maximum(current_h, 0)
                    if np.sum(current_h) > 0:
                        current_h = current_h / np.sum(current_h)
            except Exception:
                pass
        
        s_candidate = _score_raw(current_h)
        if s_candidate > best_score:
            best_score = s_candidate
            best_h = current_h

    # One last check with the best found so far using a very tight SLSQP
    if time.time() < deadline - 2:
        try:
            h_start = best_h / np.sum(best_h)
            res = minimize(
                neg_obj_raw, 
                h_start, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if res.success:
                h_candidate = res.x
                h_candidate = np.maximum(h_candidate, 0)
                h_candidate = h_candidate / np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
