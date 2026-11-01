import numpy as np
import time
from scipy.optimize import minimize

N = 400
N_FFT = 2 * N

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    F = np.fft.rfft(h, n=N_FFT)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=N_FFT)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Sidelobes are ac[1:N] (positive lags) and ac[N+1:] (negative lags)
    # Due to symmetry ac[k] = ac[N_FFT-k], we only need ac[1:N]
    return np.max(np.abs(ac[1:N]))

def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val == 0:
        return 0.0
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _spectral_objective_and_gradient(p, beta=50.0):
    """
    Objective: Optimize power spectrum p (real, non-negative) to minimize
    soft max of autocorrelation sidelobes.
    """
    # p is the power spectrum of length N_FFT//2 + 1
    # ac = irfft(p)
    ac = np.fft.irfft(p, n=N_FFT)
    
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft max
    max_val = np.max(sidelobes_abs)
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient calculation
    # d obj / d ac[k] = w_k * sign(ac[k])
    # w_k = beta * exp_shifted[k-1] / sum_exp
    # ac[t] = (1/N_FFT) * (p[0] + 2*sum_{m=1}^{N_FFT/2-1} p[m] cos(2*pi*m*t/N_FFT) + p[N_FFT/2] * (-1)^t)
    
    # Let f[k] = w_k * sign(ac[k]) for k=1..N-1
    # grad_p[m] = (2/N_FFT) * sum_{k=1}^{N-1} f[k] * cos(2*pi*m*k/N_FFT) for m=1..N_FFT/2-1
    # grad_p[0] = (1/N_FFT) * sum_{k=1}^{N-1} f[k]
    # grad_p[N_FFT/2] = (1/N_FFT) * sum_{k=1}^{N-1} f[k] * (-1)^k
    
    f = np.zeros(N)
    # Avoid division by zero if sum_exp is 0 (should not happen with beta>0)
    if sum_exp > 0:
        weights = beta * exp_shifted / sum_exp
        f[1:N] = weights * np.sign(ac[1:N])
    
    # Use FFT to compute the cosine transform efficiently
    # f_padded has f[k] at k and N_FFT-k
    f_padded = np.zeros(N_FFT)
    f_padded[1:N] = f[1:N]
    f_padded[N_FFT-N+1:N_FFT] = f[1:N][::-1]
    
    F_f = np.fft.rfft(f_padded)
    
    grad_p = np.zeros_like(p)
    if len(p) > 0:
        grad_p[0] = (1.0 / N_FFT) * F_f[0].real
    if len(p) > 1:
        # m=1 to N_FFT/2 - 1
        grad_p[1:-1] = (2.0 / N_FFT) * F_f[1:-1].real
    if len(p) > N_FFT // 2:
        # Nyquist term
        alt = np.ones(N)
        alt[1::2] = -1.0
        grad_p[N_FFT // 2] = (1.0 / N_FFT) * np.sum(f[1:N] * alt[1:N])
        
    return obj, grad_p

def _phase_retrieval(mag_spec, num_iters=50):
    """
    Gerchberg-Saxton algorithm to find a real signal h with |FFT(h)| ~ mag_spec.
    """
    # Initialize with random phases
    phase = np.random.uniform(0, 2 * np.pi, size=mag_spec.shape)
    phase[0] = 0.0
    if mag_spec.size > N_FFT // 2:
        phase[N_FFT // 2] = 0.0
        
    F_complex = mag_spec * np.exp(1j * phase)
    
    # Make full spectrum
    F_full = np.zeros(N_FFT, dtype=complex)
    F_full[:mag_spec.size] = F_complex
    for m in range(1, N_FFT // 2):
        F_full[N_FFT - m] = np.conj(F_full[m])
    if N_FFT % 2 == 0:
        F_full[N_FFT // 2] = np.conj(F_full[N_FFT // 2])
        
    h = np.fft.ifft(F_full).real
    
    for _ in range(num_iters):
        # Enforce real-valued time domain (already real)
        # Enforce magnitude constraint in frequency domain
        F_current = np.fft.rfft(h, n=N_FFT)
        
        # Update magnitude
        F_current_mag = np.abs(F_current)
        F_current_phase = np.angle(F_current)
        
        # Combine desired mag with current phase
        F_new = mag_spec * np.exp(1j * F_current_phase)
        
        # Make full spectrum
        F_full_new = np.zeros(N_FFT, dtype=complex)
        F_full_new[:F_new.size] = F_new
        for m in range(1, N_FFT // 2):
            F_full_new[N_FFT - m] = np.conj(F_full_new[m])
        if N_FFT % 2 == 0:
            F_full_new[N_FFT // 2] = np.conj(F_full_new[N_FFT // 2])
            
        h = np.fft.ifft(F_full_new).real
        
    return h

def _local_search_steepest(h, time_limit=10.0):
    """
    Steepest descent local search.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        best_delta = 0
        best_idx = -1
        best_h_new = None
        
        # Try flipping each bit
        for i in range(N):
            h[i] = -h[i]
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            delta = current_val - new_val
            h[i] = -h[i] # revert
            
            if delta > best_delta:
                best_delta = delta
                best_idx = i
                best_h_new = h.copy()
                best_h_new[i] = -best_h_new[i]
                best_ac = ac_new
                best_val = new_val
                
        if best_idx != -1:
            h = best_h_new
            ac = best_ac
            current_val = best_val
            improved = True
            
    return h, current_val

def _local_search_sa(h, time_limit=10.0):
    """
    Simulated Annealing local search.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    T = 0.5 # Initial temperature
    T_end = 0.01
    
    while time.time() - start_time < time_limit:
        # Cool down
        T = max(T_end, T * 0.99)
        
        i = np.random.randint(N)
        h[i] = -h[i]
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        delta = current_val - new_val
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            current_val = new_val
            ac = ac_new
        else:
            h[i] = -h[i] # revert
            # ac remains old
            
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    n_fft = 2 * N
    p_size = n_fft // 2 + 1
    
    # Phase 1: Spectral Shaping Optimization
    num_spectral_starts = 10
    betas = [20.0, 50.0, 100.0]
    
    for i in range(num_spectral_starts):
        if time.time() >= deadline - 20.0: # Reserve time for local search
            break
            
        # Initialize power spectrum
        if i == 0:
            p0 = np.ones(p_size)
        elif i == 1:
            freqs = np.linspace(0, 1, p_size)
            p0 = 1.0 + 0.5 * np.cos(np.pi * freqs)
        elif i == 2:
            freqs = np.linspace(0, 1, p_size)
            p0 = 1.0 + 0.5 * np.sin(np.pi * freqs)
        else:
            p0 = np.random.uniform(0.5, 1.5, size=p_size)
        
        beta = betas[i % len(betas)]
        
        def obj_grad(p, beta=beta):
            return _spectral_objective_and_gradient(p, beta)
        
        try:
            res = minimize(obj_grad, p0, method='L-BFGS-B', 
                           bounds=[(0.0, None)] * p_size,
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20})
        except Exception:
            continue
        
        p_opt = np.maximum(res.x, 0.0)
        mag_spec = np.sqrt(p_opt)
        
        # Phase Retrieval
        # Try a few random initial phases
        for pr_iter in range(3):
            if time.time() >= deadline - 10.0:
                break
            h_cont = _phase_retrieval(mag_spec, num_iters=30)
            h_bin = np.sign(h_cont)
            zero_mask = (h_bin == 0)
            if np.any(zero_mask):
                h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
                
            ac = _compute_autocorr(h_bin)
            val = _max_sidelobe(ac)
            
            if val < best_val:
                best_val = val
                best_h = h_bin.copy()
                
    # Phase 2: Refine best solution with local search
    if best_h is not None:
        # First, steepest descent
        remaining = deadline - time.time()
        if remaining > 5.0:
            t_limit = min(10.0, remaining - 5.0)
            h_ref, val_ref = _local_search_steepest(best_h.copy(), time_limit=t_limit)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
        
        # Then, simulated annealing
        remaining = deadline - time.time()
        if remaining > 5.0:
            t_limit = min(10.0, remaining - 2.0)
            h_sa, val_sa = _local_search_sa(best_h.copy(), time_limit=t_limit)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
                
        # Final polish with steepest descent again
        remaining = deadline - time.time()
        if remaining > 2.0:
            t_limit = min(5.0, remaining - 1.0)
            h_final, val_final = _local_search_steepest(best_h.copy(), time_limit=t_limit)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final

    if best_h is None:
        return np.ones(N)
        
    return best_h
