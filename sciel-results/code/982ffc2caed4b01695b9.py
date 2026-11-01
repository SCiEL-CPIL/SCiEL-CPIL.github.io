"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s kill

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

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
    # Stable log-sum-exp
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
    # where w_k = exp(beta * (ac[k] - max)) / sum_exp
    # d|ac[k]|/dx_i = sign(ac[k]) * d(ac[k])/dx_i
    # d(ac[k])/dx_i = h[i+k] + h[i-k] (if valid)
    
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        
        # ac[k] = sum_j h[j] h[j+k]
        # d(ac[k])/dx_i = h[i+k] + h[i-k] (if valid)
        # sign(ac[k]) * d(ac[k])/dx_i
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        # Vectorized update
        # Indices i such that i+k < N => i < N-k
        # grad[0:N-k] += c_k * h[k:N]
        # Indices i such that i-k >= 0 => i >= k
        # grad[k:N] += c_k * h[0:N-k]
        
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _continuous_refine(h_init, time_budget):
    """
    Use L-BFGS-B to refine a binary/continuous vector in the continuous domain.
    Returns the best continuous solution and its binary rounded version.
    """
    if _time_left() < time_budget:
        return h_init, h_init
        
    # Ensure h_init is float
    h0 = h_init.astype(np.float64)
    
    # Try a few different betas
    best_h = h0
    best_val = float('inf')
    
    betas = [20.0, 50.0, 100.0]
    start_time = time.time()
    
    for beta in betas:
        if time.time() - start_time > time_budget - 0.5:
            break
            
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        try:
            res = minimize(obj_grad, h0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 20, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 10})
            
            # Evaluate the continuous result
            ac = _compute_autocorr(res.x)
            val = _max_sidelobe(ac)
            
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
                
        except Exception:
            continue
            
    # Round to binary for final evaluation
    h_bin = np.sign(best_h)
    zero_mask = (h_bin == 0)
    if np.any(zero_mask):
        h_bin[zero_mask] = 1.0
        
    # Evaluate binary
    ac = _compute_autocorr(h_bin)
    val_bin = _max_sidelobe(ac)
    
    # Sometimes the continuous solution rounded is worse than the original binary start
    # We return the better of the two binary candidates
    ac_init = _compute_autocorr(h_init)
    val_init = _max_sidelobe(ac_init)
    
    if val_bin < val_init:
        return h_bin, h_bin
    else:
        return h_init, h_init

def _parallel_tempering_search(initial_candidates, time_limit):
    """
    Parallel Tempering (Replica Exchange) search on the binary landscape.
    initial_candidates: list of binary arrays
    time_limit: seconds
    """
    if _time_left() < time_limit:
        return initial_candidates[0], _max_sidelobe(_compute_autocorr(initial_candidates[0]))
        
    # Setup replicas
    num_replicas = min(6, len(initial_candidates))
    if num_replicas < 2:
        # Pad with randoms if needed
        while len(initial_candidates) < 2:
            initial_candidates.append(np.random.choice([-1.0, 1.0], size=N))
            
    replicas = []
    for i in range(num_replicas):
        replicas.append(initial_candidates[i].astype(np.float64).copy())
        
    # Compute initial energies
    energies = []
    for r in replicas:
        ac = _compute_autocorr(r)
        energies.append(_max_sidelobe(ac))
        
    # Temperatures: Logarithmic spacing from high to low
    # T_high should allow most flips, T_low should accept only improvements
    T_high = 0.1
    T_low = 1e-5
    temps = np.geomspace(T_high, T_low, num_replicas)
    
    best_h = replicas[0].copy()
    best_val = min(energies)
    if best_val == energies[0]:
        best_h = replicas[0].copy()
    else:
        idx_min = np.argmin(energies)
        best_h = replicas[idx_min].copy()
        best_val = energies[idx_min]
        
    start_time = time.time()
    swap_interval = 5
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Update each replica (Metropolis step)
        for i in range(num_replicas):
            T = temps[i]
            h = replicas[i]
            E = energies[i]
            
            # Flip a random bit
            idx = np.random.randint(N)
            old_val = h[idx]
            h[idx] = -old_val
            
            # Compute new energy
            ac = _compute_autocorr(h)
            new_E = _max_sidelobe(ac)
            
            delta_E = new_E - E
            
            # Metropolis acceptance
            if delta_E < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta_E / T > 50:
                    accept = False
                else:
                    accept = np.random.rand() < np.exp(-delta_E / T)
                    
            if accept:
                energies[i] = new_E
                # Update global best
                if new_E < best_val:
                    best_val = new_E
                    best_h = h.copy()
            else:
                h[idx] = old_val
                
        # 2. Replica Exchange (Swap)
        if iteration % swap_interval == 0:
            # Try swapping adjacent replicas
            # We can try all adjacent pairs or just one random pair
            # Let's try a random adjacent pair to save time
            i = np.random.randint(num_replicas - 1)
            j = i + 1
            
            E_i = energies[i]
            E_j = energies[j]
            T_i = temps[i]
            T_j = temps[j]
            
            # Delta for swap: (E_j - E_i) * (1/T_i - 1/T_j)
            # Accept with prob exp(-Delta)
            delta_swap = (E_j - E_i) * (1.0/T_i - 1.0/T_j)
            
            if delta_swap < 0:
                swap = True
            else:
                if delta_swap > 50:
                    swap = False
                else:
                    swap = np.random.rand() < np.exp(-delta_swap)
                    
            if swap:
                replicas[i], replicas[j] = replicas[j], replicas[i]
                energies[i], energies[j] = energies[j], energies[i]
                
                # Update best if the swapped-in replica is better
                # Note: best_h is a copy, so we don't need to update it here unless 
                # the replica that moved to a lower temp index was the best, but 
                # we track best_h globally. The energies list is updated.
                # We should check if the new replica at i or j is the best.
                if energies[i] < best_val:
                    best_val = energies[i]
                    best_h = replicas[i].copy()
                if energies[j] < best_val:
                    best_val = energies[j]
                    best_h = replicas[j].copy()

    return best_h, best_val

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    candidates.append(alt)
    
    # 3. Random
    rng = np.random.default_rng(42)
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 4. Blocky (random signs in blocks of 10)
    blocks = rng.choice([-1.0, 1.0], size=N//10)
    candidates.append(np.repeat(blocks, 10))
    
    # 5. Another random
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Phase 2: Continuous Refinement for each candidate
    # Allocate ~30 seconds for this phase
    time_for_refine = 30.0
    time_per_candidate = time_for_refine / len(candidates)
    
    refined_candidates = []
    for h in candidates:
        if _time_left() < 10.0:
            break
        h_ref, h_bin = _continuous_refine(h, time_per_candidate)
        refined_candidates.append(h_bin)
        
    # Phase 3: Parallel Tempering Discrete Search
    # Use the refined candidates as initial replicas
    # Allocate remaining time minus safety margin
    time_for_pt = _time_left() - 5.0
    
    if time_for_pt > 1.0:
        best_h, best_val = _parallel_tempering_search(refined_candidates, time_for_pt)
    else:
        # Fallback: simple local search on best candidate
        best_h = refined_candidates[0]
        ac = _compute_autocorr(best_h)
        best_val = _max_sidelobe(ac)
        
    return best_h
# EVOLVE-BLOCK-END
