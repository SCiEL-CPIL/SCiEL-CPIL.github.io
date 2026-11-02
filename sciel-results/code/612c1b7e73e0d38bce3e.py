import numpy as np
import time

N = 400
DELTA = 1.0 / (2 * N)  # width of each interval = (1/2)/400 = 1/800

def construct():
    start = time.time()
    deadline = start + 100.0  # safety margin before 120s
    
    # Start with a random ±1 sequence
    best_seq = None
    best_score = float('inf')
    
    # Try a few random sequences
    rng = np.random.default_rng(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        h = rng.choice([-1.0, 1.0], size=N)
        
        # Compute discrete autoconvolution
        # ac[k] = sum_i h[i] * h[i-k] for valid i
        # Using FFT for speed
        H = np.fft.fft(h, n=2*N)
        ac = np.real(np.fft.ifft(H * H.conj()))
        # ac[k] for k=0..2N-1, but we want the circular autoconvolution
        # The aperiodic autoconvolution is:
        # ac_aper[k] for k = -(N-1)..(N-1)
        # Using the FFT result: ac_circ[k] = sum_i h[i] h[(i-k) mod N]
        # We need aperiodic, so let's compute directly or use zero-padding
        
        # Actually, let me compute it properly with zero-padding
        h_pad = np.zeros(2*N)
        h_pad[:N] = h
        H_pad = np.fft.fft(h_pad, n=4*N)
        ac_aper = np.real(np.fft.ifft(H_pad * H_pad.conj()))
        # ac_aper[k] for k=0..4N-1, the aperiodic autoconvolution is in ac_aper[0..N-1] and ac_aper[4N-(N-1)..4N-1]
        # ac_aper[k] = sum_{i} h_pad[i] h_pad[i-k] for the linear (aperiodic) case
        # For k in 0..N-1: ac_aper[k] = sum_{i=k}^{N-1} h[i] h[i-k]
        # For k in N..2N-1: should be 0
        # For k in 2N..4N-1: ac_aper[k] = sum_{i=0}^{N-1-k+2N} ... 
        
        # Simpler: just take the relevant part
        # ac_aper[0] = sum h[i]^2 = N
        # ac_aper[k] for k=1..N-1 = sum_{i=k}^{N-1} h[i]*h[i-k]
        # ac_aper[4N-k] for k=1..N-1 = sum_{i=0}^{N-1-k} h[i]*h[i+k]
        
        # The max absolute value over non-zero lags:
        ac_nonzero = np.concatenate([ac_aper[1:N], ac_aper[4*N-(N-1):4*N]])
        max_ac = np.max(np.abs(ac_nonzero))
        
        # The continuous autoconvolution value is max_ac * DELTA
        # But what's the exact metric? Let's assume C3 = max_ac * DELTA
        score = max_ac * DELTA
        
        if score < best_score:
            best_score = score
            best_seq = h.copy()
    
    if best_seq is None:
        best_seq = np.ones(N)
    
    return best_seq
