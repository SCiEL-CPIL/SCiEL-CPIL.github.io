from scipy.optimize import minimize
import numpy as np
import time

N = 100

def _compute_correlations(h):
    """Compute C_k for all k in full correlation mode."""
    # np.correlate(h, 1-h, mode='full')
    # h has length N, 1-h has length N.
    # Result length 2N-1.
    # C_k corresponds to shift k.
    # In np.correlate(a, v, 'full'), output[k] = sum(a[i] * v[i+k]) 
    # Wait, np.correlate documentation:
    # c_k = sum_n a_{n+k} * conj(v_n)
    # For real: sum a[n+k] v[n].
    # We want sum h[i] (1-h[i+k]).
    # Let a = h, v = 1-h.
    # c_k = sum h[i+k] (1-h[i]).
    # This is correlation of h and 1-h.
    # The max is symmetric in h and 1-h?
    # max(corr(h, 1-h)) == max(corr(1-h, h)).
    # So direction doesn't matter for the max value.
    return np.correlate(h, 1.0 - h, mode="full")

def _smooth_objective(h, tau):
    corr = _compute_correlations(h)
    # Soft max
    # To avoid overflow, subtract max
    max_corr = np.max(corr)
    if max_corr > 1e6: return 1e6
    # Log sum exp
    lse = np.log(np.sum(np.exp((corr - max_corr) / tau))) + max_corr / tau
    return tau * lse

def _exact_objective(h):
    corr = _compute_correlations(h)
    return np.max(corr)

def _project_simplex(y, target_sum):
    """
    Project y onto {h : sum(h) = target_sum, 0 <= h_i <= 1}
    """
    # y is the point to project
    # We solve for nu: sum(clip(y_i - nu, 0, 1)) = target_sum
    
    # Bisection on nu
    # Range of nu: [min(y) - 1, max(y) + 1]
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(y - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    h_final = np.clip(y - (lo + hi) / 2.0, 0.0, 1.0)
    # Correction for floating point
    diff = target_sum - np.sum(h_final)
    if abs(diff) > 1e-9:
        h_final += diff / N
        h_final = np.clip(h_final, 0.0, 1.0)
    return h_final

def _pgd_step(h, tau, lr):
    """
    Perform one Projected Gradient Descent step on the smooth objective.
    """
    corr = _compute_correlations(h)
    max_corr = np.max(corr)
    # Probabilities
    exp_corr = np.exp((corr - max_corr) / tau)
    probs = exp_corr / np.sum(exp_corr)
    
    # Gradient of S wrt h
    # S = tau * log(sum(exp(C_k/tau)))
    # dS/dh_i = sum_k probs[k] * dC_k/dh_i
    # C_k = sum_j h[j] (1-h[j+k])
    # dC_k/dh_i = (1-h[i+k]) - h[i-k] (with wrap-around? No, correlation is linear shift, no wrap)
    # We need to handle indices carefully.
    
    # Let's compute dC/dh matrix or just the vector.
    # grad[i] = sum_k probs[k] * ( (1-h[i+k]) if i+k < N else 0 - h[i-k] if i-k >= 0 else 0 )
    
    # This is a convolution of probs with (1-h) and h.
    # term1[k] = 1-h[k+k] ? No.
    # Let's just loop or use correlate.
    
    # dC_k/dh_i = 1 if i = j and (i+k) = j+k? 
    # C_k = sum_j h_j - h_j h_{j+k}
    # dC_k/dh_i = 1 (from h_i) - (d/dh_i sum_j h_j h_{j+k})
    # = 1 - ( (1-h_{i+k}) if i is the 'j' in h_j AND i+k is in range
    #           + (-h_{i-k}) if i is the 'j+k' in h_{j+k} AND i-k is in range )
    # Wait, derivative of h_j h_{j+k} wrt h_i:
    # If i = j: term is h_{i+k}.
    # If i = j+k: term is h_{i-k}.
    # So dC_k/dh_i = 1 - ( (1-h_{i+k}) if valid_j + (-h_{i-k}) if valid_jk )?
    # No, C_k = sum h_j - sum h_j h_{j+k}.
    # dC_k/dh_i = delta_{j,i} - (delta_{j,i} h_{i+k} + h_{i-k} delta_{j+k,i})
    # = 1 - h_{i+k} (if i+k < N) - h_{i-k} (if i-k >= 0) + (correction if both apply? No, distinct terms)
    # Actually, if i is in the sum, the term is - (h_{i+k} + h_{i-k})?
    # Let's verify.
    # C_k = h_0(1-h_k) + h_1(1-h_{k+1}) + ...
    # dC_k/dh_0 = 1 - h_k.
    # dC_k/dh_k = -h_0.
    # So dC_k/dh_i = 1 (if i is in 0..N-1-k? No, i is just index in h)
    # The sum is over j=0..N-1 such that j+k < N.
    # So j ranges 0..N-1-k.
    # If i <= N-1-k, dC_k/dh_i = 1 - h_{i+k}.
    # If i > N-1-k, then i is not a 'j'.
    # But i could be a 'j+k'. j = i-k.
    # If i-k >= 0 and i-k <= N-1-k (which is always true if i<=N-1), then i is a 'j+k'.
    # So if i-k >= 0, dC_k/dh_i has contribution -h_{i-k}.
    
    # So:
    # grad[i] = sum_k probs[k] * ( 
    #   (1 - h[i+k]) if i <= N-1-k else 0 
    #   - h[i-k] if i-k >= 0 else 0 
    # )
    
    # This can be computed efficiently.
    # Let P = probs.
    # Term A_i = sum_k P_k (1-h_{i+k}) for valid k.
    # Term B_i = sum_k P_k h_{i-k} for valid k.
    # grad = A - B.
    
    # A_i = sum_k P_k - sum_k P_k h_{i+k}
    # sum_k P_k = 1.
    # So A_i = 1 - (P * h)[i]?
    # Correlation of P and h.
    # (P * h)[i] = sum_k P_k h_{i+k}.
    # This is np.correlate(P, h, mode='full')?
    # P has length 2N-1. h has length N.
    # We only care about i in 0..N-1.
    
    # Let's use np.correlate.
    # corr_Ph = np.correlate(P, h, mode='full')
    # This gives sum P[n] h[n+k].
    # We need sum P_k h_{i+k}.
    # This is convolution of P and h?
    # Convolution (P * h)[i] = sum P[k] h[i-k].
    # Correlation (P ⋆ h)[i] = sum P[k] h[i+k].
    
    # So A_i = 1 - (P ⋆ h)[i].
    # B_i = (P * h)[i] (Convolution).
    
    # Note: P is defined on k from -(N-1) to N-1.
    # np.correlate(P, h) will handle the shifts.
    
    # We need to extract the relevant part for i=0..N-1.
    # Length of P is 2N-1. Length of h is N.
    # Full correlation length 3N-2.
    
    # Let's just compute the gradient explicitly for N=100, it's cheap enough.
    grad = np.zeros(N)
    for i in range(N):
        g = 0.0
        for k in range(-(N-1), N):
            if probs[N-1+k] == 0: continue # Optimization
            pk = probs[N-1+k]
            # Term 1: 1 - h[i+k] if 0 <= i+k < N
            if 0 <= i + k < N:
                g += pk * (1.0 - h[i+k])
            # Term 2: -h[i-k] if 0 <= i-k < N
            if 0 <= i - k < N:
                g += pk * (-h[i-k])
        grad[i] = g
        
    # Step
    h_new = h - lr * grad
    h_new = _project_simplex(h_new, N/2.0)
    return h_new

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_h = np.full(N, 0.5)
    best_val = _exact_objective(best_h)
    
    def try_update(h_cand):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_cand)):
            return
        h_cand = np.clip(h_cand, 0.0, 1.0)
        diff = N/2.0 - np.sum(h_cand)
        if abs(diff) > 1e-9:
            h_cand += diff / N
            h_cand = np.clip(h_cand, 0.0, 1.0)
            
        val = _exact_objective(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()

    # Initializations
    np.random.seed(42)
    inits = [
        np.full(N, 0.5),
        np.linspace(0, 1, N),
        np.linspace(1, 0, N),
        np.random.uniform(0, 1, N),
        np.where(np.arange(N) % 2 == 0, 0.8, 0.2),
        np.where(np.arange(N) % 2 == 1, 0.8, 0.2),
        np.tile([0.0, 0.5, 1.0, 0.5], 25),
        np.tile([0.0, 0.0, 1.0, 1.0, 0.5], 20),
    ]
    
    # Add sinusoidal
    for f in [1, 2, 3, 4, 5]:
        h_s = 0.5 + 0.4 * np.cos(2 * np.pi * f * np.arange(N) / N)
        inits.append(h_s)
        
    # Process inits
    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        h = np.clip(h_init, 0, 1)
        diff = N/2.0 - np.sum(h)
        h += diff/N
        h = np.clip(h, 0, 1)
        
        # PGD with annealing
        taus = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
        lrs = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for tau, lr in zip(taus, lrs):
            # Backtracking line search or fixed steps?
            # Let's do a few steps with fixed lr, checking for improvement in smooth obj
            curr_smooth = _smooth_objective(h, tau)
            
            for step in range(50):
                if time.time() > deadline - 5:
                    break
                h_next = _pgd_step(h, tau, lr)
                next_smooth = _smooth_objective(h_next, tau)
                if next_smooth < curr_smooth:
                    h = h_next
                    curr_smooth = next_smooth
                else:
                    # Reduce lr
                    lr *= 0.5
                    if lr < 1e-6:
                        break

        # Polish with SLSQP on exact objective
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _exact_objective, 
                    h, 
                    method='SLSQP', 
                    bounds=[(0.0, 1.0)]*N, 
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                try_update(res.x)
            except:
                pass
        
        try_update(h)

    # Final global polish if time remains
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _exact_objective, 
                best_h, 
                method='SLSQP', 
                bounds=[(0.0, 1.0)]*N, 
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            try_update(res.x)
        except:
            pass

    return best_h
