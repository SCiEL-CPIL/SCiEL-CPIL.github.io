"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(x):
    """Compute the ratio: max|autocorr| / sum(x)^2, but since sum(x)=0 for ±1,
    we use max|autocorr| / sum(x^2) = max|autocorr| / N as the effective ratio."""
    # autocorrelation via FFT
    n = len(x)
    fft_size = 1
    while fft_size < 2*n - 1:
        fft_size *= 2
    X = np.fft.rfft(x, fft_size)
    ac = np.fft.irfft(X * np.conj(X), fft_size)[:n]
    peak = np.max(np.abs(ac))
    s2 = np.sum(x**2)
    return peak / s2

def construct():
    deadline = time.time() + 140.0  # safety margin
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Spectral phase retrieval: fix magnitude spectrum, optimize phase
    # For ±1 sequences, the magnitude spectrum has a specific structure
    # We'll use iterative projection: FFT -> enforce flat magnitude -> IFFT -> project to ±1
    
    rng = np.random.default_rng(42)
    
    # Try multiple initializations
    inits = []
    # Random ±1
    inits.append(rng.choice([-1.0, 1.0], size=N))
    # Alternating
    alt = np.ones(N)
    alt[1::2] = -1.0
    inits.append(alt)
    # Multi-frequency
    for k in [1, 3, 5, 7, 11, 13]:
        t = np.arange(N)
        inits.append(np.sign(np.cos(2*np.pi*k*t/N)))
    
    for x0 in inits:
        if time.time() > deadline - 10:
            break
        x = x0.copy()
        
        # Spectral phase retrieval iterations
        for it in range(5000):
            if time.time() > deadline - 5:
                break
            
            # FFT
            fft_size = 1
            while fft_size < 2*N - 1:
                fft_size *= 2
            X = np.fft.rfft(x, fft_size)
            
            # Enforce flat magnitude (all frequencies equal)
            mag = np.abs(X)
            mag[mag < 1e-12] = 1.0
            flat_mag = np.ones_like(mag)
            phase = np.angle(X)
            X_new = flat_mag * np.exp(1j * phase)
            
            # IFFT
            x_new = np.fft.irfft(X_new, fft_size)[:N]
            
            # Project to ±1
            x_new = np.sign(x_new)
            x_new[x_new == 0] = 1.0
            
            # Check improvement
            new_score = _score(x_new)
            if new_score < best_score:
                best_score = new_score
                best_x = x_new.copy()
            
            x = x_new
        
        # Local search / polish on best found so far
        if time.time() > deadline - 5:
            break
        
        # Flip-based local search
        x_cur = best_x.copy()
        cur_score = best_score
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                idx = rng.integers(0, N)
                x_trial = x_cur.copy()
                x_trial[idx] = -x_trial[idx]
                trial_score = _score(x_trial)
                if trial_score < cur_score:
                    cur_score = trial_score
                    x_cur = x_trial
                    best_x = x_trial.copy()
                    best_score = cur_score
                    improved = True
    
    return best_x
# EVOLVE-BLOCK-END
