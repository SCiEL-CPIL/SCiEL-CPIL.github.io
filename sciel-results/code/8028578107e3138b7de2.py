"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _max_sidelobe(f):
    """Compute the maximum absolute autocorrelation value for non-zero lags, normalized by N."""
    n = len(f)
    # Next power of 2 >= 2n for efficient FFT
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    
    # Use rfft for real input
    F = np.fft.rfft(f, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
    # acf[0] is the DC component (sum of squares), we want max |acf[k]| for k=1..n-1
    # Note: due to circular convolution in FFT, we only get positive lags 0..n-1 directly from irfft slice
    # However, for real signals, autocorrelation is symmetric.
    # The irfft output length is nfft//2 + 1. Slicing [:n] gives lags 0..n-1.
    # We need to exclude lag 0.
    return np.max(np.abs(acf[1:])) / n

def _generate_band_limited_signs(rng, n, num_sines=5, max_freq=10):
    """
    Generate a sign pattern based on a sum of low-frequency sine waves.
    This creates structured, smooth transitions rather than pure noise.
    """
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.zeros(n)
    
    for _ in range(num_sines):
        # Random low frequency (integer cycles to ensure periodicity or near-periodicity)
        # Using integer frequencies helps avoid beating patterns that might be bad, 
        # but random phases decorrelate them.
        freq = rng.integers(1, max_freq + 1)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.0)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
        
    # Convert to signs
    # Threshold at 0
    signs = np.sign(signal)
    # Handle zeros (rare but possible)
    signs[signs == 0] = 1.0
    return signs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Search over structured (band-limited) initializations
    best_f = None
    best_score = float('inf')
    
    num_structured_starts = 5000
    for i in range(num_structured_starts):
        if time.time() > deadline - 10.0: # Leave time for local search
            break
            
        # Vary the complexity of the signal
        # More sines = more complex, potentially better local minima escape
        num_sines = rng.integers(2, 8)
        max_freq = rng.integers(2, 15)
        
        f = _generate_band_limited_signs(rng, N, num_sines=num_sines, max_freq=max_freq)
        score = _max_sidelobe(f)
        
        if score < best_score:
            best_score = score
            best_f = f.copy()
            
    # Fallback if no structured signal was better (unlikely, but safe)
    if best_f is None:
        best_f = rng.choice([-1.0, 1.0], size=N)
        best_score = _max_sidelobe(best_f)
        
    # Phase 2: Local Search (Bit Flipping) to polish the best structured signal
    # We will run a greedy local search on the best candidate.
    # We can also try local search on a few other top candidates if time permits.
    
    # Keep a list of top candidates from Phase 1? 
    # For simplicity and speed, let's just polish the single best one found.
    # Actually, polishing the top 5 distinct patterns might be better.
    # Let's just polish the best one first, and if time remains, try random restarts of local search from scratch?
    # No, local search from random starts is what we replaced. 
    # Let's do a few passes of local search on the best_f.
    
    current_f = best_f.copy()
    current_score = best_score
    
    # Greedy local search
    # We can speed this up by pre-calculating the FFT once? 
    # No, flipping one bit changes the FFT significantly. 
    # But we can use the property that acf is real and symmetric.
    # For now, the FFT approach is O(N log N) which is very fast.
    
    improved = True
    while improved and (time.time() < deadline - 5.0):
        improved = False
        # Try flipping each bit
        # Random order to avoid bias
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 5.0:
                break
            
            current_f[i] *= -1.0
            new_score = _max_sidelobe(current_f)
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
                if current_score < best_score - 1e-12:
                    best_score = current_score
                    best_f = current_f.copy()
            else:
                current_f[i] *= -1.0
                
    # If we have significant time left, try a second round of structured starts with different parameters
    if time.time() < deadline - 10.0:
        # Maybe try higher frequency components?
        for i in range(2000):
            if time.time() > deadline - 5.0:
                break
            num_sines = rng.integers(5, 15)
            max_freq = rng.integers(10, 30)
            f = _generate_band_limited_signs(rng, N, num_sines=num_sines, max_freq=max_freq)
            score = _max_sidelobe(f)
            if score < best_score:
                best_score = score
                best_f = f.copy()
                
                # Quick local search on this new best
                current_f = best_f.copy()
                current_score = best_score
                improved = True
                while improved and (time.time() < deadline - 2.0):
                    improved = False
                    order = rng.permutation(N)
                    for i in order:
                        if time.time() > deadline - 2.0:
                            break
                        current_f[i] *= -1.0
                        new_score = _max_sidelobe(current_f)
                        if new_score < current_score - 1e-12:
                            current_score = new_score
                            improved = True
                            if current_score < best_score - 1e-12:
                                best_score = current_score
                                best_f = current_f.copy()
                        else:
                            current_f[i] *= -1.0

    return best_f

# EVOLVE-BLOCK-END
