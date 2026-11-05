"""Rewritten program: minimize autoconvolution upper bound via fast autocorrelation updates."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_initial_autocorr(s):
    """
    Compute aperiodic autocorrelation array R for a sequence s of +/-1.
    R[k] = sum_{j=0}^{N-1-k} s[j] * s[j+k] for k = 0..N-1.
    """
    n = len(s)
    # Use FFT for initial calculation for speed, though direct sum is O(N^2) which is fine for N=400
    # But FFT is O(N log N). Let's use FFT for initialization.
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    S = np.fft.rfft(s, n=fft_size)
    acf = np.fft.irfft(np.abs(S) ** 2, n=fft_size)
    # Aperiodic autocorrelation is the first N points of the linear convolution of s and reverse(s)
    # The FFT method above computes circular autocorrelation.
    # For aperiodic, we need linear convolution.
    # acf_circular[k] = sum s[i] s[i+k] (circular).
    # Since s is zero-padded to fft_size, and fft_size >= 2N-1, the first N lags are correct aperiodic autocorrelations.
    return acf[:n]

def _update_autocorr(R, s, i):
    """
    Return the new autocorrelation array if s[i] is flipped.
    Does not modify R or s in place, returns a copy or updated values.
    Actually, for efficiency in the search loop, we should compute the delta.
    """
    n = len(s)
    # R'[k] = R[k] - 2 * s[i] * (s[i+k] if i+k < n else 0) - 2 * s[i] * (s[i-k] if i-k >= 0 else 0)
    # Note: s[i] changes to -s[i].
    # Term 1: j=i in sum for lag k -> s[i] s[i+k]. New term: -s[i] s[i+k]. Diff: -2 s[i] s[i+k].
    # Term 2: j=i-k in sum for lag k -> s[i-k] s[i]. New term: s[i-k] (-s[i]). Diff: -2 s[i] s[i-k].
    
    new_R = R.copy()
    si = s[i]
    
    # Update lags where i+k < n
    # k ranges from 0 to n-1-i
    # new_R[k] -= 2 * si * s[i+k]
    if i < n:
        ks = np.arange(n - i)
        new_R[ks] -= 2 * si * s[i:i+n-i]
        
    # Update lags where i-k >= 0
    # k ranges from 0 to i
    # new_R[k] -= 2 * si * s[i-k]
    if i >= 0:
        ks = np.arange(i + 1)
        new_R[ks] -= 2 * si * s[i::-1]
        
    return new_R

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Initialize with a Chirp sequence.
    2. Compute initial autocorrelation.
    3. Use Simulated Annealing with fast O(N) autocorrelation updates.
    """
    deadline = time.time() + 140  # Safety margin
    
    # 1. Initialization
    # Chirp: s_i = sign(cos(2*pi*i^2/N))
    i_idx = np.arange(N)
    phase = 2 * np.pi * (i_idx ** 2) / N
    s = np.sign(np.cos(phase))
    # Ensure no zeros (cos can be 0)
    s[s == 0] = 1.0
    
    # 2. Compute initial Autocorrelation
    R = _compute_initial_autocorr(s)
    # The score is max |R[1:N]|
    current_peak = float(np.max(np.abs(R[1:])))
    best_peak = current_peak
    best_s = s.copy()
    best_R = R.copy()
    
    # 3. Simulated Annealing
    T = 1.0  # Temperature
    cooling = 0.999
    start_time = time.time()
    
    while time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        
        # Calculate new R
        new_R = _update_autocorr(R, s, i)
        
        # Calculate new peak
        new_peak = float(np.max(np.abs(new_R[1:])))
        
        delta = new_peak - current_peak
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            # Accept move
            s[i] *= -1.0
            R = new_R
            current_peak = new_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_s = s.copy()
                best_R = R.copy()
        
        # Cool down
        T *= cooling
        
        # Optional: Restart if stuck for a long time?
        # For now, just continue.
        
    return best_s
# EVOLVE-BLOCK-END
