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
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _spectral_objective_and_gradient(p, beta=50.0):
    """
    Objective: Optimize power spectrum p (real, non-negative) to minimize
    soft max of autocorrelation sidelobes.
    
    p is the real-valued power spectrum of length N_fft//2 + 1.
    The autocorrelation is the inverse FFT of p.
    We optimize p directly in the frequency domain.
    
    Returns (objective_value, gradient_wrt_p)
    """
    n_fft = 2 * N
    # p has length n_fft//2 + 1 (rfft output size)
    # The autocorrelation ac = ifft(p) where p is the power spectrum
    
    # Compute autocorrelation from power spectrum
    ac = np.fft.irfft(p, n=n_fft)
    
    # Sidelobes are ac[1:N] (and by symmetry ac[N+1:] = ac[N:0:-1])
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft max
    max_val = np.max(sidelobes_abs)
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient: d obj / d p
    # ac = ifft(p), so d ac[k] / d p[m] involves the FFT matrix
    # For rfft/irfft: ac = (1/n_fft) * ifft(p_extended)
    # where p_extended is the full spectrum with conjugate symmetry
    
    # Let's compute gradient analytically
    # ac[t] = sum_m p[m] * exp(2*pi*i*m*t/n_fft) / n_fft (for real case)
    # Actually, irfft(p, n) gives ac[t] = (2/n) * sum_{m=1}^{n/2-1} p[m] cos(2*pi*m*t/n) + p[0]/n + p[n/2]/n * (-1)^t
    
    # More precisely for n_fft even:
    # ac[t] = (1/n_fft) * (p[0] + 2*sum_{m=1}^{n_fft/2-1} p[m] cos(2*pi*m*t/n_fft) + p[n_fft/2] * (-1)^t)
    
    # So d ac[t] / d p[m] = (2/n_fft) * cos(2*pi*m*t/n_fft) for 1 <= m < n_fft/2
    #                        (1/n_fft) for m = 0
    #                        (1/n_fft) * (-1)^t for m = n_fft/2
    
    # Gradient of soft max w.r.t. ac[1:N]:
    # w_k = beta * exp_shifted[k-1] / sum_exp for k=1..N-1
    # sign_k = sign(ac[k])
    # d obj / d ac[k] = w_k * sign_k
    
    # Then d obj / d p[m] = sum_{k=1}^{N-1} (d obj / d ac[k]) * (d ac[k] / d p[m])
    
    # Build gradient
    grad_p = np.zeros_like(p)
    
    # Precompute cos terms for efficiency
    # We need sum_{k=1}^{N-1} w_k * sign_k * cos(2*pi*m*k/n_fft) for each m
    
    # This is essentially an FFT-like operation
    # Let f[k] = w_k * sign_k for k=1..N-1
    # Then grad_p[m] = (2/n_fft) * sum_{k=1}^{N-1} f[k] * cos(2*pi*m*k/n_fft)
    
    f = np.zeros(N)
    f[1:N] = beta * exp_shifted / sum_exp * np.sign(ac[1:N])
    
    # Compute the cosine transform
    # grad_p[m] = (2/n_fft) * real(fft(f_padded)[m]) for m=1..n_fft/2-1
    # where f_padded has f[k] at positions k and n_fft-k
    
    f_padded = np.zeros(n_fft)
    f_padded[1:N] = f[1:N]
    f_padded[n_fft-N+1:n_fft] = f[1:N][::-1]
    
    F_f = np.fft.rfft(f_padded)
    
    # grad_p[m] = (2/n_fft) * F_f[m] for m=1..n_fft/2-1
    # grad_p[0] = (1/n_fft) * sum f[k] = (1/n_fft) * F_f[0]
    # grad_p[n_fft/2] = (1/n_fft) * sum f[k] * (-1)^k
    
    if len(p) > 0:
        grad_p[0] = (1.0 / n_fft) * F_f[0].real
    if len(p) > 1:
        grad_p[1:-1] = (2.0 / n_fft) * F_f[1:-1].real
    if len(p) > n_fft // 2:
        # For the Nyquist term
        alt = np.ones(N)
        alt[1::2] = -1.0
        grad_p[n_fft // 2] = (1.0 / n_fft) * np.sum(f[1:N] * alt[1:N])
    
    return obj, grad_p

def _local_search(h, max_iterations=10000, time_limit=5.0):
    """
    Fast local search using FFT for evaluation.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
        else:
            # Revert
            h[i] = old_h_i
            # ac remains unchanged (old ac)
            
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Spectral Shaping Optimization
    # Optimize the power spectrum directly to minimize sidelobe peak
    n_fft = 2 * N
    p_size = n_fft // 2 + 1
    
    num_spectral_starts = 8
    betas = [20.0, 50.0, 100.0]
    
    for i in range(num_spectral_starts):
        if time.time() >= deadline - 25.0:  # Reserve time for local search
            break
            
        # Initialize power spectrum
        # Start with a reasonable initial guess
        if i == 0:
            # Flat spectrum (corresponds to random signal)
            p0 = np.ones(p_size)
        elif i == 1:
            # Low-pass shaped
            freqs = np.linspace(0, 1, p_size)
            p0 = 1.0 + 0.5 * np.cos(np.pi * freqs)
        elif i == 2:
            # High-pass shaped
            freqs = np.linspace(0, 1, p_size)
            p0 = 1.0 + 0.5 * np.sin(np.pi * freqs)
        else:
            # Random positive spectrum
            p0 = np.random.uniform(0.5, 1.5, size=p_size)
        
        beta = betas[i % len(betas)]
        
        # Define objective with fixed beta
        def obj_grad(p, beta=beta):
            return _spectral_objective_and_gradient(p, beta)
        
        # Use L-BFGS-B with non-negativity constraint on p
        try:
            res = minimize(obj_grad, p0, method='L-BFGS-B', 
                           bounds=[(0.0, None)] * p_size,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
        except Exception:
            continue
        
        # Get the optimized power spectrum
        p_opt = np.maximum(res.x, 0.0)
        
        # Compute the corresponding signal
        # The autocorrelation is ac = ifft(p_opt)
        # We need to find h such that |FFT(h)|^2 ≈ p_opt
        # This is a phase retrieval problem
        # Simple heuristic: take the square root and use random phases
        
        # Compute the desired magnitude spectrum
        # p_opt is the power spectrum, so |F[m]| = sqrt(p_opt[m])
        mag_spec = np.sqrt(p_opt)
        
        # Create a complex spectrum with random phases
        # But we need h to be real, so F must have conjugate symmetry
        # F[m] = mag_spec[m] * exp(i * phase[m])
        # For real h: F[n_fft-m] = conj(F[m])
        
        # Use a simple approach: random phases for m=1..n_fft/2-1
        # F[0] and F[n_fft/2] are real
        
        phase = np.random.uniform(0, 2 * np.pi, size=p_size)
        phase[0] = 0.0  # DC term is real
        if p_size > n_fft // 2:
            phase[n_fft // 2] = 0.0  # Nyquist is real
        
        F_complex = mag_spec * np.exp(1j * phase)
        
        # Make it conjugate symmetric
        F_full = np.zeros(n_fft, dtype=complex)
        F_full[:p_size] = F_complex
        for m in range(1, n_fft // 2):
            F_full[n_fft - m] = np.conj(F_full[m])
        if n_fft % 2 == 0:
            F_full[n_fft // 2] = np.conj(F_full[n_fft // 2])
        
        # Inverse FFT to get h
        h_cont = np.fft.ifft(F_full).real
        
        # Threshold to ±1
        h_bin = np.sign(h_cont)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
        
        # Evaluate binary solution
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
    
    # Phase 2: Refine best solution with local search
    if best_h is not None and time.time() < deadline - 15.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=10000, time_limit=10.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 3: Final Discrete Search
    remaining_time = deadline - time.time()
    if remaining_time > 3.0 and best_h is not None:
        h_final, val_final = _local_search(best_h.copy(), max_iterations=100000, time_limit=remaining_time - 2.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
