import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0

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

def _freq_phase_objective(phases, beta=50.0):
    """
    Objective: minimize peak sidelobe by optimizing frequency domain phases.
    
    Args:
        phases: array of length N/2+1 (rfft frequencies)
        beta: soft-max parameter
    
    Returns:
        tuple: (objective_value, gradient)
    """
    # Reconstruct complex spectrum from phases (unit magnitude)
    # phases[0] is DC, phases[1:N/2] are positive freqs
    n_fft = 2 * N
    half = N // 2
    
    # Build full complex spectrum from phases
    # For real signal, F[k] = F[N-k]*, so we define first half
    F = np.zeros(n_fft // 2 + 1, dtype=complex)
    F[0] = np.exp(1j * phases[0])  # DC
    if half >= 1:
        F[1:half+1] = np.exp(1j * phases[1:half+1])
    
    # Extend to full spectrum for irfft
    # F[k] for k > N/2 is conjugate of F[N-k]
    F_full = np.zeros(n_fft // 2 + 1, dtype=complex)
    F_full[0] = F[0]
    if half >= 1:
        F_full[1:half+1] = F[1:half+1]
    if n_fft // 2 > half:
        F_full[half+1:n_fft//2+1] = np.conj(F_full[n_fft - (half+1):n_fft - 1][::-1])
    
    # Compute autocorrelation from power spectrum
    power = F_full * np.conj(F_full)
    ac = np.fft.irfft(power, n=n_fft)
    
    # Compute soft-max of sidelobes
    sidelobes_abs = np.abs(ac[1:N])
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros_like(phases)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient computation
    # d(ac[k])/d(phases[m]) involves chain rule through FFT
    # This is complex; use numerical gradient approximation for stability
    grad = np.zeros_like(phases)
    eps = 1e-5
    for i in range(len(phases)):
        phases_plus = phases.copy()
        phases_plus[i] += eps
        F_plus = np.zeros(n_fft // 2 + 1, dtype=complex)
        F_plus[0] = np.exp(1j * phases_plus[0])
        if half >= 1:
            F_plus[1:half+1] = np.exp(1j * phases_plus[1:half+1])
        if n_fft // 2 > half:
            F_plus[half+1:n_fft//2+1] = np.conj(F_plus[n_fft - (half+1):n_fft - 1][::-1])
        power_plus = F_plus * np.conj(F_plus)
        ac_plus = np.fft.irfft(power_plus, n=n_fft)
        sidelobes_plus = np.abs(ac_plus[1:N])
        max_plus = np.max(sidelobes_plus)
        shifted_plus = beta * (sidelobes_plus - max_plus)
        exp_plus = np.exp(shifted_plus)
        sum_plus = np.sum(exp_plus)
        obj_plus = max_plus + np.log(sum_plus) / beta
        
        phases_minus = phases.copy()
        phases_minus[i] -= eps
        F_minus = np.zeros(n_fft // 2 + 1, dtype=complex)
        F_minus[0] = np.exp(1j * phases_minus[0])
        if half >= 1:
            F_minus[1:half+1] = np.exp(1j * phases_minus[1:half+1])
        if n_fft // 2 > half:
            F_minus[half+1:n_fft//2+1] = np.conj(F_minus[n_fft - (half+1):n_fft - 1][::-1])
        power_minus = F_minus * np.conj(F_minus)
        ac_minus = np.fft.irfft(power_minus, n=n_fft)
        sidelobes_minus = np.abs(ac_minus[1:N])
        max_minus = np.max(sidelobes_minus)
        shifted_minus = beta * (sidelobes_minus - max_minus)
        exp_minus = np.exp(shifted_minus)
        sum_minus = np.sum(exp_minus)
        obj_minus = max_minus + np.log(sum_minus) / beta
        
        grad[i] = (obj_plus - obj_minus) / (2 * eps)
    
    return obj, grad

def _freq_to_binary(phases):
    """
    Map frequency phases to a binary sequence via IFFT and thresholding.
    """
    n_fft = 2 * N
    half = N // 2
    
    F = np.zeros(n_fft // 2 + 1, dtype=complex)
    F[0] = np.exp(1j * phases[0])
    if half >= 1:
        F[1:half+1] = np.exp(1j * phases[1:half+1])
    if n_fft // 2 > half:
        F[half+1:n_fft//2+1] = np.conj(F[n_fft - (half+1):n_fft - 1][::-1])
    
    # Inverse FFT to get time domain
    h_complex = np.fft.irfft(F, n=n_fft)
    
    # Take first N samples and threshold
    h_real = h_complex[:N].real
    h_bin = np.sign(h_real)
    zero_mask = (h_bin == 0)
    if np.any(zero_mask):
        h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
    
    return h_bin

def _local_search(h, max_iterations=10000, time_limit=5.0, use_pair_flips=True):
    """
    Fast local search using FFT for evaluation with simulated annealing and pair flips.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    T0 = max(current_val * 0.15, 0.01)
    cooling_rate = 0.9995
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type: single flip or pair flip
        if use_pair_flips and it % 3 == 0:
            # Pair flip: flip two random bits
            i, j = np.random.choice(N, size=2, replace=False)
            old_i, old_j = h[i], h[j]
            h[i] = -h[i]
            h[j] = -h[j]
            
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            
            delta = new_val - current_val
            T = T0 * (cooling_rate ** it)
            if delta < 0 or (T > 1e-15 and np.random.random() < np.exp(-delta / T)):
                current_val = new_val
                ac = ac_new
            else:
                h[i] = old_i
                h[j] = old_j
        else:
            # Single flip
            i = np.random.randint(N)
            old_h_i = h[i]
            h[i] = -h[i]
            
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            
            delta = new_val - current_val
            T = T0 * (cooling_rate ** it)
            if delta < 0 or (T > 1e-15 and np.random.random() < np.exp(-delta / T)):
                current_val = new_val
                ac = ac_new
            else:
                h[i] = old_h_i
                
    return h, current_val

def _refine_discrete(h, time_limit=2.0):
    """
    Quick discrete refinement using greedy improvement.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            h[i] = -h[i]
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            if new_val < current_val:
                current_val = new_val
                ac = ac_new
                improved = True
            else:
                h[i] = -h[i]
                
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Frequency-domain phase retrieval
    num_freq_starts = 8
    half = N // 2
    
    for i in range(num_freq_starts):
        if time.time() >= deadline - 30.0:
            break
        
        # Initialize phases randomly
        phases0 = np.random.uniform(0, 2*np.pi, size=half+1)
        
        try:
            res = minimize(_freq_phase_objective, phases0, method='BFGS',
                           options={'maxiter': 50, 'gtol': 1e-4})
        except Exception:
            continue
        
        # Map to binary
        h_bin = _freq_to_binary(res.x)
        
        # Evaluate binary solution
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
        
        # Quick greedy refinement
        if best_h is not None and time.time() < deadline - 25.0:
            h_ref, val_ref = _refine_discrete(best_h.copy(), time_limit=1.0)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Phase 2: Local search with simulated annealing and pair flips
    if best_h is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            h_final, val_final = _local_search(best_h.copy(), max_iterations=500000, time_limit=remaining_time - 5.0, use_pair_flips=True)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
    
    # Phase 3: Final greedy polish
    if best_h is not None and time.time() < deadline - 2.0:
        h_polish, val_polish = _refine_discrete(best_h.copy(), time_limit=1.5)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
    
    if best_h is None:
        return np.ones(N)
        
    return best_h
