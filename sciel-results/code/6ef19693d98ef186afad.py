import numpy as np
import time

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()

def _time_left():
    return _DEADLINE - (time.monotonic() - _T0)

def _score(x):
    """
    Calculate the metric: max_{k != 0} |c_k| / (sum x)^2
    Assumes x is a 1D numpy array.
    """
    S = np.sum(x)
    if S == 0:
        return np.inf
    n_fft = 2 * N
    F = np.fft.fft(x, n=n_fft)
    P = F * np.conj(F)
    c = np.fft.ifft(P).real
    sidelobes = c[1:N]
    max_sid = np.max(np.abs(sidelobes))
    return max_sid / (S * S)

def _autoconv_max_abs(x):
    """Compute max absolute autoconvolution excluding lag 0."""
    n_fft = 2 * N
    F = np.fft.fft(x, n=n_fft)
    P = F * np.conj(F)
    c = np.fft.ifft(P).real
    return np.max(np.abs(c[1:N]))

def _local_search(x, max_steps=5000):
    """Greedy local search with sign flips."""
    current_x = x.copy()
    current_score = _score(current_x)
    best_x = current_x.copy()
    best_score = current_score
    
    for _ in range(max_steps):
        if _time_left() < 2.0:
            break
        idx = np.random.randint(N)
        current_x[idx] *= -1
        new_score = _score(current_x)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            current_x[idx] *= -1
    
    return best_x, best_score

def construct():
    # Spectral design approach:
    # 1. Design a magnitude profile in frequency domain to minimize peak sidelobes
    # 2. Project to signs
    # 3. Local search refinement
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)
    
    if _time_left() < 5.0:
        return best_x
    
    # Try multiple spectral designs with different target magnitudes
    # The idea: choose |F(k)| such that the inverse FFT has low peak sidelobes
    
    # Design 1: Uniform magnitude (all ones) - baseline
    # Design 2: Tapered magnitude (Hamming-like window in freq domain)
    # Design 3: Optimized magnitude via iterative water-filling
    
    # Start with a few candidate initializations
    candidates = []
    
    # Candidate 1: All ones
    candidates.append(np.ones(N, dtype=np.float64))
    
    # Candidate 2: Random signs
    for _ in range(3):
        candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Candidate 3: Spectral design - flat spectrum with optimized phase
    # We want |F(k)| to be constant, but the phase should be chosen to minimize sidelobes
    # Start with random phases and iterate
    
    for trial in range(5):
        if _time_left() < 5.0:
            break
        
        # Initialize with random phase
        freq_idx = np.arange(N)
        # Only use first N/2 + 1 frequencies (real signal)
        half_n = N // 2 + 1
        random_phase = np.random.uniform(0, 2 * np.pi, size=half_n)
        
        # Construct frequency domain
        F = np.zeros(N, dtype=np.complex128)
        # Set magnitude to 1 for all frequencies
        F[:half_n] = np.exp(1j * random_phase[:half_n])
        # Ensure Hermitian symmetry for real signal
        for k in range(1, half_n):
            F[N - k] = np.conj(F[k])
        
        # Inverse FFT to get time domain
        x_trial = np.fft.ifft(F).real
        
        # Project to signs
        x_trial = np.sign(x_trial)
        x_trial[x_trial == 0] = 1.0  # Handle zeros
        
        # Local search refinement
        x_refined, score_refined = _local_search(x_trial, max_steps=3000)
        
        if score_refined < best_score:
            best_score = score_refined
            best_x = x_refined.copy()
    
    # Final polish with more local search steps
    if _time_left() > 10.0:
        best_x, best_score = _local_search(best_x, max_steps=10000)
    
    return best_x
