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
    
    # New approach: Softmax parameterization in frequency domain + simulated annealing
    # Key insight: Use log-amplitude parameterization with softmax to ensure non-negativity
    # and smooth optimization landscape. Simulated annealing helps escape local minima.
    
    rng = np.random.default_rng(42)
    
    # Initialize: broad Gaussian in frequency domain (smooth in time)
    k = np.arange(n)
    # Start with a smooth, low-frequency-dominated spectrum
    log_amp = -k**2 / (n/3)**2
    log_amp[0] = 0.0
    phase = np.zeros(n)
    
    def params_to_f(log_amp, phase):
        """Convert log-amplitude and phase to non-negative time-domain f."""
        # Softmax for non-negativity: f_i = exp(log_amp_i) ensures f_i > 0
        # But we need to be careful: we want the DFT coefficients to have these amplitudes
        # Actually, let's parameterize differently:
        # Let F_k = exp(log_amp_k) * exp(i*phase_k)
        # Then f = ifft(F), and we project to non-negative
        # But projection creates non-smoothness.
        # Alternative: directly parameterize f_i = exp(x_i) where x_i are free params
        # This guarantees non-negativity and smoothness.
        
        # Let's use direct time-domain parameterization: f_i = exp(x_i)
        # This is cleaner and avoids FFT round-trip issues.
        # But the current interface expects us to work in a way that's efficient.
        
        # Actually, let me reconsider. The problem is to choose f_i >= 0.
        # The most direct parameterization is f_i = exp(x_i) for free x_i.
        # Then we minimize max(autoconv(f)) / (sum f)^2.
        
        # But we're in frequency domain in the current code. Let me switch to time-domain
        # parameterization for cleanliness.
        
        F = np.exp(log_amp) * np.exp(1j * phase)
        f_raw = np.fft.ifft(F).real
        f = np.maximum(f_raw, 0.0)
        # Normalize to have sum = n (for stability)
        s = np.sum(f)
        if s > 1e-12:
            f = f * (n / s)
        return f
    
    def compute_score_from_params(log_amp, phase):
        f = params_to_f(log_amp, phase)
        return _compute_score(f), f
    
    # Initial evaluation
    best_score, best_f = compute_score_from_params(log_amp, phase)
    best_log_amp = log_amp.copy()
    best_phase = phase.copy()
    
    # Simulated annealing with adaptive step size
    T_init = 0.5
    T_min = 1e-6
    step_init = 0.1
    step_min = 1e-4
    
    # Phase 1: Simulated annealing for global exploration
    # Use a geometric cooling schedule
    n_sa_iter = 2000
    for it in range(n_sa_iter):
        if time.time() > deadline - 30.0:
            break
        
        # Adaptive temperature: geometric cooling
        T = T_init * (T_min / T_init) ** (it / n_sa_iter)
        
        # Adaptive step size: larger early, smaller late
        step = step_init * (step_min / step_init) ** (it / n_sa_iter)
        
        # Generate candidate: perturb log_amp and phase
        d_log_amp = rng.normal(0, step, n)
        d_phase = rng.normal(0, step, n)
        
        cand_log_amp = best_log_amp + d_log_amp
        cand_phase = best_phase + d_phase
        
        cand_score, cand_f = compute_score_from_params(cand_log_amp, cand_phase)
        
        # Accept if better or with probability exp(-delta/T)
        delta = cand_score - best_score
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / max(T, 1e-12))):
            best_log_amp = cand_log_amp
            best_phase = cand_phase
            best_score = cand_score
            best_f = cand_f
    
    # Phase 2: Directed local search targeting autoconvolution peaks
    # Use gradient-like steps in time domain
    if time.time() < deadline - 15.0:
        f = best_f.copy()
        s = np.sum(f)
        if s > 0:
            f = f * (n / s)
        best_s = _compute_score(f)
        eta = 0.05
        
        max_iter = 500
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            conv = _autoconv(f)
            cm = np.max(conv)
            
            # Build sensitivity vector: gradient of max_conv w.r.t. f
            # Use subgradient of max: weight indices near the max
            idx = np.where(conv >= cm * 0.95)[0]
            S = np.zeros(n)
            for m in idx:
                lag = m - (n - 1)
                lag_abs = abs(lag)
                if lag_abs == 0:
                    # c_0 = sum f_j^2, d/d f_j = 2*f_j
                    S += 2 * f
                else:
                    # c_lag = sum_j f_j f_{j+lag}
                    # d/d f_j = f_{j+lag} + f_{j-lag} (where defined)
                    if lag > 0:
                        for j in range(n - lag):
                            S[j] += f[j + lag]
                            S[j + lag] += f[j]
                    else:
                        lag_pos = -lag
                        for j in range(n - lag_pos):
                            S[j] += f[j + lag_pos]
                            S[j + lag_pos] += f[j]
            
            # Normalize sensitivity
            sm, sd = S.mean(), S.std() + 1e-12
            nrm = np.clip((S - sm) / sd, -4.0, 4.0)
            
            # Step in direction that reduces sensitivity at peaks
            cand = f * (1.0 - eta * nrm)
            cand = np.maximum(cand, 1e-10)  # Keep strictly positive
            cs = np.sum(cand)
            if cs <= 0:
                break
            cand = cand * (n / cs)
            
            cv = _compute_score(cand)
            if cv < best_s - 1e-12:
                f = cand
                best_s = cv
                best_f = cand.copy()
                eta = min(eta * 1.03, 0.3)
            else:
                eta *= 0.9
                if eta < 1e-5:
                    break
    
    f = np.maximum(best_f, 1e-10)
    if np.sum(f) < 1e-12:
        f = np.ones(n)
    return [float(x) for x in f]
