import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _init_sine(freq, phase):
    """Initialize sign vector from a low-frequency sine wave."""
    x = np.linspace(-0.25, 0.25, N)
    wave = np.sin(2 * np.pi * freq * x + phase)
    s = np.where(wave >= 0, 1.0, -1.0)
    return s

def _local_search(s, A, current_val, deadline):
    """Perform local search with exact delta updates for ACF."""
    improved = True
    passes = 0
    max_passes = 50
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                return s, A, current_val
            
            old_si = s[i]
            
            # Compute delta for A when flipping s[i]
            # A_new[k] = A[k] - 2 * old_si * (s[i+k] + s[i-k])
            # where s[j] = 0 if j < 0 or j >= N
            
            delta = np.zeros(N)
            
            # Part 1: k where i+k < N, so k in [0, N-1-i]
            # s[i+k] for k=0..N-1-i is s[i:N]
            if N - 1 - i >= 0:
                delta[:N - i] += -2.0 * old_si * s[i:N]
            
            # Part 2: k where i-k >= 0, so k in [0, i]
            # s[i-k] for k=0..i is s[i::-1]
            if i >= 0:
                delta[:i+1] += -2.0 * old_si * s[i::-1]
            
            # Apply delta to get A_new
            A_new = A + delta
            
            # Compute new max abs (excluding k=0)
            val_new = float(np.max(np.abs(A_new[1:])))
            
            if val_new < current_val - 1e-12:
                # Accept the flip
                s[i] = -old_si
                A = A_new
                current_val = val_new
                improved = True
            # else: revert (s[i] unchanged, A unchanged)
    
    return s, A, current_val

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Try different sine wave initializations
    freqs = [1, 2, 3, 4, 5, 6, 8, 10]
    
    while time.time() < deadline:
        # Choose a frequency and phase
        freq = np.random.choice(freqs)
        phase = np.random.uniform(0, 2 * np.pi)
        
        s = _init_sine(freq, phase)
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search
        s, A, current_val = _local_search(s, A, current_val, deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
        
        # Also try random initialization occasionally
        if np.random.random() < 0.3:
            s = np.random.randint(0, 2, size=N) * 2 - 1
            A = _calc_acf(s)
            current_val = _max_abs_acf_from_array(A)
            
            s, A, current_val = _local_search(s, A, current_val, deadline)
            
            if current_val < best_val:
                best_val = current_val
                best_seq = s.copy()

    return best_seq
