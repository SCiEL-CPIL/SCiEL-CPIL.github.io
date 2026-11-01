"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


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


def _c3_upper_bound(h):
    """
    Compute the C3 upper bound: max_k |conv(h,h)[k]| / (sum |h_i|)^2
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    max_ac = np.max(np.abs(ac[1:N]))
    l1_norm = np.sum(np.abs(h))
    return max_ac / (l1_norm ** 2)


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


def _spectral_objective_and_gradient(log_power, beta=50.0):
    """
    Optimize in frequency domain: minimize peak of inverse FFT of power spectrum.
    log_power is log of power spectrum (real, length N//2+1 for rfft).
    """
    # Reconstruct power spectrum
    power = np.exp(log_power)
    
    # Compute autocorrelation from power spectrum
    # ac[k] = IDFT(power)[k]
    n_fft = 2 * N
    ac = np.fft.irfft(power, n=n_fft)
    
    # Get sidelobes
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft max
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros_like(log_power)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient w.r.t. log_power
    # Weight for each sidelobe k
    weights = beta * exp_shifted / sum_exp
    
    # Gradient computation using vectorized approach
    m_indices = np.arange(len(power))
    
    # For each sidelobe k, compute contribution to gradient
    # d(ac[k])/d(power[m]) = 2*cos(2*pi*k*m/n_fft)/n_fft for m>0, 1/n_fft for m=0
    grad_power = np.zeros(len(power))
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        # Vectorized: for each m, contribution is c_k * 2*cos(2*pi*k*m/n_fft)/n_fft
        cos_terms = 2 * np.cos(2 * np.pi * k * m_indices / n_fft) / n_fft
        cos_terms[0] = 1.0 / n_fft  # DC term
        
        grad_power += c_k * cos_terms
    
    # d(obj)/d(log_power[m]) = d(obj)/d(power[m]) * power[m]
    grad_log_power = grad_power * power
    
    return obj, grad_log_power


def _local_search(h, max_iterations=10000, time_limit=5.0, use_sa=False, T0=0.01, multi_flip_prob=0.3):
    """
    Fast local search using FFT for evaluation.
    Optionally uses Simulated Annealing (SA) to escape local minima.
    Supports multi-bit flips for better exploration.
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

    n_fft = 2 * N
    rng = np.random.default_rng(42)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Decide how many bits to flip
        if rng.random() < multi_flip_prob:
            n_flips = rng.integers(2, 5)
        else:
            n_flips = 1
        
        # Select random positions to flip
        flip_indices = rng.choice(N, size=n_flips, replace=False)
        old_vals = h[flip_indices].copy()
        h[flip_indices] = -h[flip_indices]
        
        # Recompute autocorrelation efficiently
        F_new = np.fft.rfft(h, n=n_fft)
        power_new = F_new * np.conj(F_new)
        ac_new = np.fft.irfft(power_new, n=n_fft)
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
                delta = new_val - current_val
                if delta < 0:
                    accepted = True
                else:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        accepted = True
            
            if accepted:
                current_val = new_val
                ac = ac_new
            else:
                # Revert
                h[flip_indices] = old_vals
        
        if T > 1e-9:
            T *= 0.9995  # Cooling factor

    return best_h, best_val


def _spectral_to_time(log_power, n_iterations=100):
    """
    Convert optimized power spectrum back to time-domain signal.
    Uses phase retrieval via iterative projection with multiple restarts.
    """
    power = np.exp(log_power)
    n_fft = 2 * N
    
    best_h = None
    best_val = float('inf')
    
    # Try multiple random phase initializations
    for restart in range(5):
        rng = np.random.default_rng(12345 + restart)
        phase = rng.uniform(0, 2*np.pi, size=len(power))
        H = power * np.exp(1j * phase)
        
        # Iterative projection: FFT -> threshold to ±1 -> IFFT -> keep magnitude
        h_est = np.zeros(N)
        for _ in range(n_iterations):
            # Get time domain
            h_cont = np.fft.irfft(H, n=n_fft)[:N]
            
            # Threshold to ±1
            h_bin = np.sign(h_cont)
            zero_mask = (h_bin == 0)
            if np.any(zero_mask):
                h_bin[zero_mask] = rng.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
            # Get frequency domain
            H_new = np.fft.rfft(h_bin, n=n_fft)
            
            # Keep magnitude from H, phase from H_new
            mag_H = np.abs(H)
            phase_H_new = np.angle(H_new)
            H = mag_H * np.exp(1j * phase_H_new)
            
            h_est = h_bin
        
        # Evaluate this candidate
        ac = _compute_autocorr(h_est)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_est.copy()
    
    if best_h is None:
        return np.ones(N)
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Spectral Shaping Optimization
    num_spectral_starts = 6
    betas = [20.0, 50.0, 100.0, 200.0, 50.0, 100.0]
    
    for i in range(num_spectral_starts):
        if time.time() >= deadline - 40.0:  # Reserve 40s for final polish
            break
            
        rng = np.random.default_rng(42 + i)
        beta = betas[i % len(betas)]
        
        # Initialize log_power: start with flat spectrum (white noise)
        # log_power[0] is DC, log_power[1:N//2] are positive frequencies
        log_power0 = np.zeros(N // 2 + 1)
        # Add small random perturbation
        log_power0 += rng.normal(0, 0.1, size=N // 2 + 1)
        
        # Define objective
        def obj_grad(log_power, beta=beta):
            return _spectral_objective_and_gradient(log_power, beta)
        
        # Use L-BFGS-B
        try:
            res = minimize(obj_grad, log_power0, method='L-BFGS-B', 
                           options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
        except Exception:
            continue
        
        # Map to time domain
        h_candidate = _spectral_to_time(res.x, n_iterations=80)
        
        # Evaluate
        ac = _compute_autocorr(h_candidate)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()
            
        # Quick local search refinement
        if time.time() < deadline - 30.0:
            h_ref, val_ref = _local_search(h_candidate.copy(), max_iterations=2000, time_limit=2.0, multi_flip_prob=0.4)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref

    # Phase 2: Refine best solution with local search
    if best_h is not None and time.time() < deadline - 20.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=10000, time_limit=8.0, multi_flip_prob=0.3)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 3: Final Discrete Search with Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_h is not None:
        T0 = max(0.01, best_val * 0.05) 
        h_final, val_final = _local_search(best_h.copy(), max_iterations=200000, time_limit=remaining_time - 2.0, use_sa=True, T0=T0, multi_flip_prob=0.2)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
