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
    
    # Spectral approach: parameterize by DFT amplitudes and phases
    # f = inverse_dft(amp * exp(i*phase))
    # Enforce non-negativity via projection: f_i = max(0, raw_f_i)
    # Minimize max(f*f) / (sum f)^2
    
    # Initialize with a broad Gaussian-like shape in frequency domain
    # Low frequencies dominant -> smooth, flat-ish time domain
    k = np.arange(n)
    amp = np.exp(-k**2 / (n/4)**2)
    amp[0] = 1.0
    phase = np.zeros(n)
    
    def amp_to_f(amp, phase):
        """Convert spectral params to time-domain f (non-negative)."""
        F = amp * np.exp(1j * phase)
        f_raw = np.fft.ifft(F).real
        # Projection for non-negativity
        f = np.maximum(f_raw, 0.0)
        return f
    
    def compute_from_amp(amp, phase):
        f = amp_to_f(amp, phase)
        score = _compute_score(f)
        return f, score
    
    f, best_score = compute_from_amp(amp, phase)
    best_amp = amp.copy()
    best_phase = phase.copy()
    
    # Coordinate descent on amplitude and phase
    # For each frequency bin, try small perturbations
    rng = np.random.default_rng(42)
    
    def try_perturb(amp, phase, i, is_amp, delta):
        """Try a perturbation at index i and return (new_score, new_val)."""
        if is_amp:
            test_amp = amp.copy()
            test_amp[i] = amp[i] + delta
            f_test, score_test = compute_from_amp(test_amp, phase)
            return score_test, test_amp[i]
        else:
            test_phase = phase.copy()
            test_phase[i] = phase[i] + delta
            f_test, score_test = compute_from_amp(amp, test_phase)
            return score_test, test_phase[i]
    
    while time.time() < deadline:
        improved = False
        
        # Amplitude coordinate descent
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            
            orig_amp = best_amp[i]
            best_local = best_score
            best_local_amp = orig_amp
            
            # Try a few perturbations
            for delta in [0.05, -0.05, 0.2, -0.2, 0.5, -0.5]:
                score_test, new_val = try_perturb(best_amp, best_phase, i, True, delta)
                if score_test < best_local:
                    best_local = score_test
                    best_local_amp = new_val
            
            if best_local < best_score - 1e-10:
                best_amp[i] = best_local_amp
                best_score = best_local
                improved = True
        
        # Phase coordinate descent
        for i in range(1, n):  # Skip DC (phase 0)
            if time.time() > deadline - 5.0:
                break
            
            orig_phase = best_phase[i]
            best_local = best_score
            best_local_phase = orig_phase
            
            for delta in [0.05, -0.05, 0.2, -0.2, 0.5, -0.5]:
                score_test, new_val = try_perturb(best_amp, best_phase, i, False, delta)
                if score_test < best_local:
                    best_local = score_test
                    best_local_phase = new_val
            
            if best_local < best_score - 1e-10:
                best_phase[i] = best_local_phase
                best_score = best_local
                improved = True
        
        if not improved:
            # Try random restart with perturbation
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                test_amp = best_amp + rng.normal(0, 0.05, n)
                test_amp = np.maximum(test_amp, 0)
                test_phase = best_phase + rng.normal(0, 0.05, n)
                f_test, score_test = compute_from_amp(test_amp, test_phase)
                if score_test < best_score:
                    best_amp = test_amp
                    best_phase = test_phase
                    best_score = score_test
                    improved = True
            if not improved:
                break
    
    f = amp_to_f(best_amp, best_phase)
    
    # Final polish: local search on time-domain f using gradient-like direction
    # based on autoconvolution peaks (inspired by inspiration 1)
    if time.time() < deadline - 10.0:
        f = np.asarray(f, dtype=np.float64)
        s = np.sum(f)
        if s > 0:
            f = f * (n / s)
        best_f = f.copy()
        best_s = _compute_score(f)
        eta = 0.03
        
        while time.time() < deadline - 5.0:
            conv = _autoconv(f)
            cm = np.max(conv)
            # Find indices where conv is close to max
            idx = np.where(conv >= cm * 0.99)[0]
            # Build sensitivity vector S
            S = np.zeros(n)
            for m in idx:
                # m is index in conv (length 2n-1), lag = m - (n-1)
                lag = m - (n - 1)
                if lag >= 0:
                    # c_lag = sum_{j=0}^{n-1-lag} f[j] f[j+lag]
                    # d(c_lag)/df[j] = f[j+lag] for j < n-lag, and f[j-lag] for j >= lag
                    for j in range(n - lag):
                        S[j] += f[j + lag]
                        S[j + lag] += f[j]
                else:
                    # Same as positive lag -lag by symmetry
                    lag_pos = -lag
                    for j in range(n - lag_pos):
                        S[j] += f[j + lag_pos]
                        S[j + lag_pos] += f[j]
            
            sm, sd = S.mean(), S.std() + 1e-12
            nrm = np.clip((S - sm) / sd, -3.0, 3.0)
            cand = f * (1.0 - eta * nrm)
            cand = np.maximum(cand, 0.0)
            cs = np.sum(cand)
            if cs <= 0:
                break
            cand = cand * (n / cs)
            cv = _compute_score(cand)
            if cv < best_s - 1e-12:
                f = cand
                best_s = cv
                best_f = cand.copy()
                eta = min(eta * 1.05, 0.2)
            else:
                eta *= 0.85
                if eta < 1e-4:
                    break
    
    f = np.maximum(f, 0.0)
    if np.sum(f) < 1e-12:
        f = np.ones(n)
    return [float(x) for x in f]
