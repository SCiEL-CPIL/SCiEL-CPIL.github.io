import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Initialize with Legendre Sequence
    # Find smallest prime P > N such that P % 4 == 1
    # For N=400, P=401 is prime and 401 % 4 == 1
    P = 401
    # Compute Legendre symbols
    # L[i] = 1 if i is QR mod P, -1 if QNR, 0 if i=0
    # We map 0 to 1 to keep values in {-1, 1}
    h = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        # Legendre symbol (i/P) = i^((P-1)/2) % P
        # If result is 1, QR. If P-1, QNR.
        res = pow(i, (P-1)//2, P)
        if res == 1:
            h[i] = 1
        else:
            h[i] = -1
    h[0] = 1 # Map 0 to 1
    
    # Compute initial autocorrelation
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    current_peak = np.max(np.abs(r[1:N]))
    
    best_h = h.copy()
    best_peak = current_peak
    
    # 2. Local Search
    # We will try to flip bits to reduce the peak.
    # To speed up, we can precompute the contribution of each bit to each lag?
    # Or just do random flips and check.
    # Given N=400, full FFT is fast.
    
    # Use a simulated annealing-like approach or simple hill climbing.
    # Since Legendre is already good, simple hill climbing should work.
    
    rng = np.random.default_rng(42)
    
    # To avoid getting stuck, we can allow some bad moves (annealing) or just restart if stuck.
    # But Legendre is a strong start.
    
    for _ in range(2000): # More iterations since start is good and we want to polish
        if time.time() > deadline:
            break
        
        # Pick a random index to flip
        i = rng.integers(N)
        h[i] *= -1
        
        # Recompute peak
        F_new = np.fft.rfft(h, n=n_fft)
        power_new = F_new * np.conj(F_new)
        r_new = np.fft.irfft(power_new, n=n_fft)
        new_peak = np.max(np.abs(r_new[1:N]))
        
        if new_peak < current_peak:
            current_peak = new_peak
            r = r_new
            if new_peak < best_peak:
                best_peak = new_peak
                best_h = h.copy()
        else:
            # Revert
            h[i] *= -1
            
        # Optional: If stuck for a while, try a multi-flip?
        # For now, keep it simple.
        
    return best_h
