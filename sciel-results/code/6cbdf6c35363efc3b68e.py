import numpy as np
import time

def _autoconv(f):
    """Compute autoconvolution (convolution f*f) using FFT.
    Returns array of length 2n-1."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # New approach: optimize in frequency domain using simulated annealing
    # Key insight: autoconvolution is related to periodogram, so frequency
    # domain optimization is natural. Use soft non-negativity to avoid
    # projection artifacts.
    
    def freq_to_f(freq_mag, freq_phase):
        """Convert frequency magnitude and phase to time-domain f with soft non-negativity."""
        # Construct complex frequency
        freq = freq_mag * np.exp(1j * freq_phase)
        # Ensure Hermitian symmetry for real f
        # freq[0] is DC (real), freq[n/2] is Nyquist (real if n even)
        if n % 2 == 0:
            freq[n//2] = np.abs(freq[n//2])
        # Enforce Hermitian symmetry: freq[n-k] = conj(freq[k])
        for k in range(1, n//2):
            freq[n-k] = np.conj(freq[k])
        
        f_raw = np.fft.ifft(freq).real
        # Soft non-negativity: use softplus to keep f >= 0 smoothly
        # f = softplus(f_raw) = log(1 + exp(f_raw))
        # But we need to scale to maintain reasonable magnitude
        f = np.log1p(np.exp(f_raw))
        # Normalize to have sum = n (standard normalization)
        s = np.sum(f)
        if s > 0:
            f = f * (n / s)
        return f
    
    def compute_score_from_freq(freq_mag, freq_phase):
        f = freq_to_f(freq_mag, freq_phase)
        return _compute_score(f)
    
    # Initialize: start with a smooth frequency profile
    k = np.arange(n)
    # Start with low-frequency dominant profile
    init_mag = np.exp(-k**2 / (2 * (n/8)**2))
    init_mag[0] = 1.0
    init_phase = np.zeros(n)
    
    best_f = freq_to_f(init_mag, init_phase)
    best_score = _compute_score(best_f)
    current_mag = init_mag.copy()
    current_phase = init_phase.copy()
    
    # Simulated annealing
    T = 0.1  # Initial temperature
    cooling = 0.9995  # Cooling rate
    T_min = 1e-4
    
    # Track best
    best_mag = current_mag.copy()
    best_phase = current_phase.copy()
    
    while time.time() < deadline - 10.0:
        if T < T_min:
            break
        
        # Generate candidate: perturb frequency magnitude and phase
        # Use adaptive perturbation based on current temperature
        sigma_mag = T * 0.5
        sigma_phase = T * 2.0
        
        # Perturb a random subset of frequency bins
        n_perturb = max(1, int(n * T * 0.1))
        idx = rng.choice(n, size=n_perturb, replace=False)
        
        cand_mag = current_mag.copy()
        cand_phase = current_phase.copy()
        
        cand_mag[idx] += rng.normal(0, sigma_mag, n_perturb)
        cand_mag[idx] = np.maximum(cand_mag[idx], 0)  # Magnitude must be non-negative
        cand_phase[idx] += rng.normal(0, sigma_phase, n_perturb)
        
        # Compute candidate score
        cand_score = compute_score_from_freq(cand_mag, cand_phase)
        
        # Metropolis acceptance
        if cand_score < best_score:
            # Always accept if better than best
            current_mag = cand_mag
            current_phase = cand_phase
            if cand_score < best_score:
                best_score = cand_score
                best_mag = cand_mag.copy()
                best_phase = cand_phase.copy()
                best_f = freq_to_f(best_mag, best_phase)
        else:
            # Accept with probability exp(-(cand_score - best_score) / T)
            delta = cand_score - best_score
            if delta < 0:
                accept = True
            else:
                accept_prob = np.exp(-delta / max(T, 1e-10))
                accept = rng.random() < accept_prob
            
            if accept:
                current_mag = cand_mag
                current_phase = cand_phase
        
        # Cool down
        T *= cooling
        
        # Periodically reheat if stuck
        if int((deadline - time.time()) % 10) == 0 and T < 0.01:
            T = max(T, 0.05)
    
    # Final: use best solution
    f = freq_to_f(best_mag, best_phase)
    
    # Local refinement in time domain
    if time.time() < deadline - 5.0:
        s = np.sum(f)
        if s > 0:
            f = f * (n / s)
        best_f_local = f.copy()
        best_s_local = _compute_score(f)
        
        # Simple local search: try small perturbations
        for iteration in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Generate random perturbation
            delta_f = rng.normal(0, 0.01, n)
            cand_f = f + delta_f
            cand_f = np.maximum(cand_f, 0)
            cs = np.sum(cand_f)
            if cs <= 0:
                continue
            cand_f = cand_f * (n / cs)
            
            cand_s = _compute_score(cand_f)
            if cand_s < best_s_local:
                f = cand_f
                best_f_local = cand_f.copy()
                best_s_local = cand_s
    
    f = np.maximum(f, 0.0)
    if np.sum(f) < 1e-12:
        f = np.ones(n)
    return [float(x) for x in f]
