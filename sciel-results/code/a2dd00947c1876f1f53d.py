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
    # Continuous autoconvolution of step function f(t) = sum h_i 1_{[x_i, x_{i+1})}(t)
    # The peak is proportional to the max of the discrete convolution of h with itself.
    # Specifically, max(f*f) = DELTA * max(conv(h,h))
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _softmax_max(v, beta):
    """
    Compute a smooth approximation to max(v) using softmax.
    v: 1D array
    beta: temperature parameter (higher = sharper max)
    """
    v_max = np.max(v)
    exp_v = np.exp(beta * (v - v_max))
    return v_max + np.log(np.sum(exp_v)) / beta

def _score_soft(h, beta):
    """
    Compute a smooth score using softmax approximation for the autoconvolution peak.
    This is differentiable with respect to h.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    if np.max(h_conv) <= 1e-12:
        return 0.0
    
    # Use softmax to approximate max
    max_conv_soft = _softmax_max(h_conv, beta)
    
    # The raw score is l2_sq / (l1 * DELTA * max_conv)
    # We want to maximize this, so minimize its negative.
    # score_soft = l2_sq / (l1 * DELTA * max_conv_soft)
    if max_conv_soft <= 1e-12:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv_soft)

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial best guess
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Parametric search for good initializations
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # 4. Random sparse initialization
    for _ in range(100):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        # Add some noise
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using L-BFGS-B on log-scale variables
    # h_i = exp(x_i)
    # Objective: Minimize - _score_soft(h, beta)
    # We will anneal beta from low to high to sharpen the max approximation.
    
    # Helper for L-BFGS-B objective
    def neg_score_soft_log(x, beta):
        h = np.exp(x)
        # Scale h to have sum 1? No, the score is scale invariant in the sense that
        # _score_raw(c*h) = _score_raw(h)/c.
        # Wait, if we use log variables, h can grow arbitrarily.
        # _score_raw(h) = L2^2 / (L1 * DELTA * MaxConv).
        # If h -> c*h, L2^2 -> c^2 L2^2, L1 -> c L1, MaxConv -> c^2 MaxConv.
        # Ratio -> (c^2 L2^2) / (c L1 * c^2 MaxConv) = (1/c) Ratio.
        # So minimizing -score is equivalent to maximizing score.
        # If we let x grow, h grows, score decreases (goes to 0).
        # If we let x shrink, h shrinks, score increases (goes to inf).
        # This is a problem for unconstrained optimization.
        # We must constrain the scale.
        # The most natural constraint is L1=1 (probability distribution).
        # Let's fix L1=1 by parameterizing h_i = exp(x_i) / sum(exp(x_j)).
        # This is the softmax parameterization.
        
        # Softmax parameterization
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        sum_exp_x = np.sum(exp_x)
        h = exp_x / sum_exp_x
        
        s = _score_soft(h, beta)
        if s <= 0:
            return 0.0
        return -s

    # We will run a few rounds of L-BFGS-B with increasing beta.
    # Start with a relatively soft beta to find the basin, then sharpen.
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    current_h = best_h / np.sum(best_h)
    # Initialize x from current_h
    # x_i = log(h_i) + log(sum(h)) = log(h_i) since sum(h)=1.
    # But we need x such that softmax(x) ~ h.
    # x = log(h) works if h > 0.
    # If any h_i is 0, we have a problem.
    # Let's ensure h_i > 0.
    current_h = np.maximum(current_h, 1e-10)
    current_h = current_h / np.sum(current_h)
    x_current = np.log(current_h)
    
    for beta in betas:
        if time.time() > deadline - 20: break
        
        try:
            res = minimize(neg_score_soft_log, x_current, args=(beta,), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                # Reconstruct h
                x_res = res.x
                x_max = np.max(x_res)
                exp_x = np.exp(x_res - x_max)
                h_res = exp_x / np.sum(exp_x)
                
                # Calculate actual raw score
                s_res = _score_raw(h_res)
                
                if s_res > best_score:
                    best_score = s_res
                    best_h = h_res
                    x_current = x_res
                else:
                    # Even if score didn't improve, update x_current to the result
                    # to stay in the basin?
                    # Actually, if score didn't improve, it might be better to keep
                    # the previous x or the one that gave the best soft score.
                    # Let's just update x_current to res.x for continuity in beta annealing.
                    x_current = x_res
                    
        except Exception:
            pass

    # Final polish with SLSQP on the raw score (non-smooth) to ensure we are at a local optimum
    # of the true objective, using the best_h found by L-BFGS-B as start.
    
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 == 0: return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0: return 0
        return -(l2_sq / (l1 * DELTA * max_conv))
        
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h} # h >= 0
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    h_start = best_h / np.sum(best_h)
    h_start = np.maximum(h_start, 1e-10)
    h_start = h_start / np.sum(h_start)
    
    try:
        res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            h_candidate = res.x
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
    except Exception:
        pass

    # Try a few random restarts with L-BFGS-B if time permits
    if time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline - 20: break
            # Random init
            h_rand = np.random.rand(N)
            h_rand /= np.sum(h_rand)
            x_rand = np.log(np.maximum(h_rand, 1e-10))
            
            # Run a short L-BFGS-B with high beta
            beta = 50.0
            try:
                res = minimize(neg_score_soft_log, x_rand, args=(beta,), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10})
                if res.success:
                    x_res = res.x
                    x_max = np.max(x_res)
                    exp_x = np.exp(x_res - x_max)
                    h_res = exp_x / np.sum(exp_x)
                    s_res = _score_raw(h_res)
                    if s_res > best_score:
                        best_score = s_res
                        best_h = h_res
            except Exception:
                pass

    return list(best_h)

# EVOLVE-BLOCK-END
