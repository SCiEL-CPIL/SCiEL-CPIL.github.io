import numpy as np
import time

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
    return np.max(np.abs(ac[1:N]))

def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val
    shifted = beta * (sidelobes - max_val)
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
    
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
        
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-12:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0, use_sa=False, T0=0.01, parallel_tempering=False):
    """
    Fast local search using incremental autocorrelation updates.
    Optionally uses Simulated Annealing (SA) or Parallel Tempering to escape local minima.
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
    
    # For parallel tempering, we maintain multiple replicas at different temperatures
    if parallel_tempering:
        num_replicas = 5
        temperatures = [0.01 * (0.5 ** i) for i in range(num_replicas)]
        # replicas: list of (h, ac, val, T)
        replicas = []
        for i in range(num_replicas):
            h_rep = h.copy()
            ac_rep = _compute_autocorr(h_rep)
            val_rep = _max_sidelobe(ac_rep)
            replicas.append([h_rep, ac_rep, val_rep, temperatures[i]])
        
        for _ in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # For each replica, try a flip
            for r in range(num_replicas):
                h_rep = replicas[r][0]
                ac_rep = replicas[r][1]
                val_rep = replicas[r][2]
                T_rep = replicas[r][3]
                
                i = np.random.randint(N)
                old_h_i = h_rep[i]
                h_rep[i] = -old_h_i
                
                # Incremental update
                delta_ac = np.zeros(2*N)
                for k in range(1, N):
                    delta = 0.0
                    if i + k < N:
                        delta -= 2.0 * old_h_i * h_rep[i + k]
                    if i - k >= 0:
                        delta -= 2.0 * h_rep[i - k] * old_h_i
                    delta_ac[k] = delta
                    delta_ac[2*N - k] = delta
                
                ac_new = ac_rep + delta_ac
                new_val = _max_sidelobe(ac_new)
                
                # Accept/reject
                accepted = False
                if T_rep > 1e-9:
                    delta_val = new_val - val_rep
                    if delta_val < 0:
                        accepted = True
                    else:
                        prob = np.exp(-delta_val / T_rep)
                        if np.random.rand() < prob:
                            accepted = True
                
                if accepted:
                    replicas[r][1] = ac_new
                    replicas[r][2] = new_val
                else:
                    h_rep[i] = old_h_i
                
                # Update best
                if replicas[r][2] < best_val:
                    best_val = replicas[r][2]
                    best_h = replicas[r][0].copy()
            
            # Swap replicas with different temperatures (Metropolis criterion)
            for r in range(num_replicas - 1):
                if np.random.rand() < 0.1:  # 10% chance to attempt swap
                    # Compute energy difference
                    delta_E = replicas[r+1][2] - replicas[r][2]
                    # Accept swap with probability min(1, exp(-delta_E / (T_high - T_low)))
                    # But since we're minimizing, and T_high > T_low:
                    # We want to swap if it helps the lower temp replica
                    # Standard PT: accept swap with prob min(1, exp(-(E1 - E2)/(T1 - T2)))
                    # where T1 > T2, E1 is energy of replica 1 (higher T), E2 of replica 2
                    # Here r+1 has lower T than r (since temperatures decrease with index)
                    # Wait, temperatures = [0.01, 0.005, 0.0025, ...] so index 0 has highest T
                    # So r has higher T than r+1
                    # delta_E = E_{r+1} - E_r
                    # Accept with prob min(1, exp(-delta_E / (T_r - T_{r+1})))
                    T_diff = replicas[r][3] - replicas[r+1][3]
                    if T_diff > 1e-10:
                        if delta_E < 0:
                            accept_swap = True
                        else:
                            accept_swap = np.random.rand() < np.exp(-delta_E / T_diff)
                    else:
                        accept_swap = False
                    
                    if accept_swap:
                        # Swap h, ac, val, T
                        h_r, ac_r, val_r, T_r = replicas[r]
                        h_rp1, ac_rp1, val_rp1, T_rp1 = replicas[r+1]
                        replicas[r] = [h_rp1, ac_rp1, val_rp1, T_r]
                        replicas[r+1] = [h_r, ac_r, val_r, T_rp1]
                        
                        if val_rp1 < best_val:
                            best_val = val_rp1
                            best_h = h_rp1.copy()
                        if val_r < best_val:
                            best_val = val_r
                            best_h = h_r.copy()
            
            # Cool down
            for r in range(num_replicas):
                if replicas[r][3] > 1e-9:
                    replicas[r][3] *= 0.999
        
        return best_h, best_val
    
    # Standard SA / Hill climbing
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        delta_ac = np.zeros(2*N)
        for k in range(1, N):
            delta = 0.0
            if i + k < N:
                delta -= 2.0 * old_h_i * h[i + k]
            if i - k >= 0:
                delta -= 2.0 * h[i - k] * old_h_i
            delta_ac[k] = delta
            delta_ac[2*N - k] = delta
        
        ac_new = ac + delta_ac
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            accepted = False
            if T > 1e-9:
                delta_val = new_val - current_val
                if delta_val < 0:
                    accepted = True
                else:
                    prob = np.exp(-delta_val / T)
                    if np.random.rand() < prob:
                        accepted = True
            
            if accepted:
                current_val = new_val
                ac = ac_new
            else:
                h[i] = old_h_i
        
        if T > 1e-9:
            T *= 0.999

    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with Smooth Objective
    num_continuous_starts = 15
    betas = [20.0, 50.0, 100.0, 200.0, 300.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 25.0:
            break
            
        x0 = np.random.choice([-1.0, 1.0], size=N)
        beta = betas[i % len(betas)]
        
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        try:
            from scipy.optimize import minimize
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
        except Exception:
            continue
        
        h_bin = np.sign(res.x)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        if best_h is not None:
            if time.time() < deadline - 10.0:
                h_ref, val_ref = _local_search(best_h.copy(), max_iterations=2000, time_limit=2.0)
                if val_ref < best_val:
                    best_val = val_ref
                    best_h = h_ref
    
    # Phase 2: Final Discrete Search with Parallel Tempering
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_h is not None:
        # Use parallel tempering for better exploration
        h_final, val_final = _local_search(best_h.copy(), max_iterations=100000, time_limit=remaining_time - 2.0, 
                                           use_sa=True, T0=0.05, parallel_tempering=True)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
