import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Due to symmetry, max |ac[1:N]| is sufficient
    return np.max(np.abs(ac[1:N]))

def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val
    # Stable log-sum-exp: max_val + log(sum(exp(x - max_val)))
    # We want a smooth max, so we use:
    # soft_max(x) = (1/beta) * log(sum(exp(beta * x)))
    # = (1/beta) * log(sum(exp(beta * (x - max_val) + beta * max_val)))
    # = max_val + (1/beta) * log(sum(exp(beta * (x - max_val))))
    
    shifted = beta * (sidelobes - max_val)
    # Clip to avoid overflow in exp (though shifted <= 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of max sidelobe.
    Gradient: Analytic.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft max value
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
        
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient of soft max w.r.t. sidelobes
    # d/dx_i (soft_max) = sum_k w_k * d|ac[k]|/dx_i
    # where w_k = exp(beta * (ac[k] - max)) / sum_exp = beta * exp_shifted[k] / sum_exp
    # Note: d|ac[k]|/dx_i = sign(ac[k]) * d(ac[k])/dx_i
    # d(ac[k])/dx_i = h[i+k] + h[i-k] (if valid)
    
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    # Only process weights that are significant
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-12:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        # Vectorized update
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0, use_sa=False, T0=0.01):
    """
    Fast local search using incremental autocorrelation updates.
    Optionally uses Simulated Annealing (SA) to escape local minima.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    if use_sa and current_val > 1e-10:
        T = T0
    else:
        T = 0.0

    # Precompute for faster evaluation
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        # Incrementally update autocorrelation
        # When we flip h[i], the change in ac[k] is:
        # ac_new[k] = ac_old[k] - 2*h[i]*(h[i+k] + h[i-k]) for valid indices
        # But since we flipped h[i], the new h[i] = -old_h_i
        # So the change is: delta_ac[k] = 2*old_h_i*(h[i+k] + h[i-k])
        # Wait, let me think more carefully.
        
        # Original: ac[k] = sum_j h[j]*h[j+k]
        # After flipping h[i]: new_h[i] = -h[i]
        # New ac[k] = sum_j new_h[j]*new_h[j+k]
        # The terms that change are those involving index i:
        # j=i: new_h[i]*new_h[i+k] = -h[i]*new_h[i+k]
        # j=i-k: new_h[i-k]*new_h[i] = new_h[i-k]*(-h[i])
        # j such that j+k=i, i.e., j=i-k: already counted
        # j such that j=i: already counted
        # Also j where j+k=i means j=i-k, and j where j=i means the term h[i]*h[i+k]
        
        # Actually, the autocorrelation is: ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
        # When we flip h[i], the affected terms are:
        # 1. j=i (if i < N-k): h[i]*h[i+k] becomes -h[i]*h[i+k]
        # 2. j=i-k (if i-k >= 0 and i-k < N-k, i.e., k <= i < N): h[i-k]*h[i] becomes h[i-k]*(-h[i])
        
        # So delta_ac[k] = -2*h[i]*h[i+k] (if i < N-k) - 2*h[i]*h[i-k] (if i >= k)
        # But note: if both conditions hold, we have both terms.
        
        # Let's compute the new autocorrelation incrementally
        delta_ac = np.zeros(2*N)
        for k in range(1, N):
            delta = 0.0
            # Term where j = i: contributes -2*h[i]*h[i+k] if i+k < N
            if i + k < N:
                delta -= 2.0 * old_h_i * h[i + k]
            # Term where j = i-k: contributes -2*h[i-k]*h[i] if i-k >= 0
            if i - k >= 0:
                delta -= 2.0 * h[i - k] * old_h_i
            delta_ac[k] = delta
            # By symmetry, delta_ac[2N-k] = delta_ac[k]
            delta_ac[2*N - k] = delta
        
        ac_new = ac + delta_ac
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            # Accept improvement
            current_val = new_val
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Check for acceptance in SA
            accepted = False
            if T > 1e-9:
                delta_val = new_val - current_val
                if delta_val < 0:
                    accepted = True
                else:
                    # Accept with probability exp(-delta / T)
                    prob = np.exp(-delta_val / T)
                    if np.random.rand() < prob:
                        accepted = True
            
            if accepted:
                current_val = new_val
                ac = ac_new
            else:
                # Revert
                h[i] = old_h_i
                # ac remains unchanged (old ac)
        
        if T > 1e-9:
            T *= 0.999  # Faster cooling

    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with Smooth Objective
    # Try a few random initializations
    num_continuous_starts = 15
    betas = [20.0, 50.0, 100.0, 200.0, 300.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 25.0:  # Reserve 25s for final polish
            break
            
        # Initialize x in [-1, 1]
        # Start near binary to help rounding
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        # Use a beta that increases over iterations or pick one
        # Let's try different betas for different starts
        beta = betas[i % len(betas)]
        
        # Define objective with fixed beta
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        # Use L-BFGS-B with more iterations
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
        except Exception:
            continue
        
        # Round to binary
        h_bin = np.sign(res.x)
        # Handle zeros
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        # Evaluate binary solution
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        # Quick local search refinement on this candidate
        if time.time() < deadline - 15.0:
            h_ref, val_ref = _local_search(h_bin.copy(), max_iterations=3000, time_limit=2.0)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref

    # Phase 2: Refine best solution with local search
    if best_h is not None and time.time() < deadline - 15.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=10000, time_limit=5.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 3: Final Discrete Search with Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_h is not None:
        # Use SA to potentially escape the local minimum found by greedy local search
        # T0 is set relative to current best value to allow meaningful jumps
        T0 = max(0.005, best_val * 0.03) 
        h_final, val_final = _local_search(best_h.copy(), max_iterations=200000, time_limit=remaining_time - 2.0, use_sa=True, T0=T0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    # Phase 4: Final exact polish pass
    if best_h is not None and time.time() < deadline - 1.0:
        # Do one more round of pure local search (no SA) to polish
        h_polish, val_polish = _local_search(best_h.copy(), max_iterations=50000, time_limit=deadline - time.time() - 1.0, use_sa=False)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
