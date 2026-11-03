import numpy as np
import time

N = 50
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def _compute_ratio(h):
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Autoconvolution: (h * h)(k) = sum_j h[j]*h[j+k] for k in [-(N-1), N-1]
    # Using FFT for efficiency
    fft_h = np.fft.fft(h, n=2*N)
    autocorr = np.fft.ifft(fft_h * np.conj(fft_h)).real
    # autocorr has length 2N, indices 0..N-1 are positive lags, N..2N-1 are negative
    sup_norm = np.max(np.abs(autocorr))
    if sup_norm == 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _compute_ratio_from_frequencies(freqs, amps, phases):
    """Compute ratio from frequency-domain representation.
    
    h[j] = |sum_k amps[k] * exp(i*(freqs[k]*j + phases[k]))|^2
    """
    N = len(freqs) if len(freqs) < 50 else 50
    j = np.arange(50)
    # Build complex signal
    z = np.zeros(50, dtype=complex)
    for k in range(len(freqs)):
        z += amps[k] * np.exp(1j * (freqs[k] * j + phases[k]))
    h = np.abs(z) ** 2
    return _compute_ratio(h)

def _optimize_frequencies(freqs, amps, phases, time_budget):
    """Optimize frequency parameters using gradient-free methods."""
    start_time = time.time()
    best_params = (freqs.copy(), amps.copy(), phases.copy())
    best_score = _compute_ratio_from_frequencies(freqs, amps, phases)
    
    n_params = len(freqs)
    total_dims = 3 * n_params  # freqs, amps, phases
    
    # Simple random restarts with local perturbations
    n_restarts = max(50, int(time_budget * 10))
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Try different strategies
        if restart == 0:
            # Start from current best
            cur_freqs = best_params[0].copy()
            cur_amps = best_params[1].copy()
            cur_phases = best_params[2].copy()
        else:
            # Random initialization
            cur_freqs = np.random.uniform(0, 2*np.pi, n_params)
            cur_amps = np.random.uniform(0.5, 2.0, n_params)
            cur_phases = np.random.uniform(0, 2*np.pi, n_params)
        
        cur_score = _compute_ratio_from_frequencies(cur_freqs, cur_amps, cur_phases)
        
        # Local search with adaptive step size
        step_size = 0.5
        for step in range(200):
            if time.time() - start_time > time_budget:
                break
            
            improved = False
            # Try random perturbations
            for _ in range(10):
                trial_freqs = cur_freqs + np.random.normal(0, step_size, n_params)
                trial_amps = cur_amps + np.random.normal(0, step_size*0.1, n_params)
                trial_phases = cur_phases + np.random.normal(0, step_size, n_params)
                
                trial_score = _compute_ratio_from_frequencies(trial_freqs, trial_amps, trial_phases)
                
                if trial_score > cur_score:
                    cur_freqs, cur_amps, cur_phases = trial_freqs, trial_amps, trial_phases
                    cur_score = trial_score
                    improved = True
                    break
            
            if not improved:
                step_size *= 0.8
            else:
                step_size *= 1.1
            
            if step_size < 0.01:
                break
        
        if cur_score > best_score:
            best_score = cur_score
            best_params = (cur_freqs.copy(), cur_amps.copy(), cur_phases.copy())
    
    return best_params, best_score

def construct():
    start_time = time.time()
    
    # Try frequency-domain optimization
    best_score = 0.0
    best_h = None
    
    # Try different numbers of frequency components
    for n_freq in [2, 3, 4, 5]:
        remaining_time = TIME_LIMIT - SAFETY_MARGIN - (time.time() - start_time)
        if remaining_time < 2.0:
            break
        
        freqs = np.random.uniform(0, 2*np.pi, n_freq)
        amps = np.ones(n_freq)
        phases = np.random.uniform(0, 2*np.pi, n_freq)
        
        params, score = _optimize_frequencies(freqs, amps, phases, remaining_time * 0.8)
        
        if score > best_score:
            best_score = score
            j = np.arange(50)
            z = np.zeros(50, dtype=complex)
            for k in range(n_freq):
                z += params[1][k] * np.exp(1j * (params[0][k] * j + params[2][k]))
            best_h = np.abs(z) ** 2
    
    # Also try the Sidon set approach as baseline
    S = [0, 1, 4, 10, 18, 23, 30]
    h_sidon = [0.0] * N
    for idx in S:
        h_sidon[idx] = 1.0 / 7.0
    sidon_score = _compute_ratio(h_sidon)
    
    if best_h is None or sidon_score > best_score:
        return h_sidon
    
    # Normalize to ensure nonnegativity and reasonable scale
    best_h = np.maximum(best_h, 0)
    return list(best_h)
