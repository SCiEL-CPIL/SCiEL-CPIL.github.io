import numpy as np
import time

N = 400

def _autoconv_max_abs(signs):
    """Compute max |autoconvolution| for lags 1..N-1"""
    # signs is array of ±1 of length N
    # autoconv at lag k: sum_{i=0}^{N-1-k} signs[i] * signs[i+k]
    # Use FFT for efficiency
    # Zero-pad to 2N
    n = len(signs)
    size = 1
    while size < 2*n:
        size *= 2
    f = np.zeros(size)
    f[:n] = signs
    F = np.fft.fft(f)
    acf = np.real(np.fft.ifft(F * np.conj(F)))
    # acf[0] is the DC term (n), acf[1..n-1] are lags 1..n-1
    # Also need negative lags but they're symmetric for real signals
    # The autoconvolution (not autocorrelation) for real signals:
    # c(k) = sum_{i} f[i] * f[i+k]
    # This is what we computed. For k > 0, acf[k] gives sum f[i]*f[i+k]
    # For k < 0, acf[k] = acf[-k] for real signals? No, autoconvolution isn't symmetric.
    # Actually for autoconvolution: c(k) = sum_i f[i] f[i+k]
    # c(-k) = sum_i f[i] f[i-k] = sum_j f[j+k] f[j] = c(k)
    # So it IS symmetric for autoconvolution (not cross-correlation)
    # So we just need lags 1..n-1
    return np.max(np.abs(acf[1:n]))

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Initialize with Thue-Morse sequence for low autocorrelation
    def thue_morse(n):
        tm = [1]
        while len(tm) < n:
            tm.extend([-x for x in tm])
        return np.array(tm[:n])
    
    signs = thue_morse(N).astype(np.float64)
    
    # Compute initial energy
    current_energy = _autoconv_max_abs(signs)
    best_signs = signs.copy()
    best_energy = current_energy
    
    # Simulated annealing with block flips
    T_start = 1.0
    T_end = 0.001
    n_iterations = 50000
    
    rng = np.random.RandomState(42)
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Temperature schedule (geometric cooling)
        T = T_start * (T_end / T_start) ** (it / n_iterations)
        
        # Choose block size
        block_size = rng.choice([4, 8, 16, 32])
        
        # Choose random start position
        start = rng.randint(0, N - block_size + 1)
        
        # Compute energy delta for flipping this block
        # New signs: flip block
        new_signs = signs.copy()
        new_signs[start:start+block_size] *= -1
        
        new_energy = _autoconv_max_abs(new_signs)
        
        # Metropolis acceptance
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
            signs = new_signs
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_signs = signs.copy()
        # else: reject, keep current
    
    return best_signs
