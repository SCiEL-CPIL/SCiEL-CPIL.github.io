"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_abs_autocorr(f):
    """Compute max |autocorrelation| for non-zero lags using FFT."""
    n = len(f)
    size = 1
    while size < 2 * n:
        size <<= 1
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.rfft(f_pad)
    acf = np.fft.irfft(F * np.conj(F), size)
    if n > 1:
        pos_lags = acf[1:n]
        neg_lags = acf[size-n+1:size]
        return float(np.max(np.abs(np.concatenate([pos_lags, neg_lags]))))
    return 0.0

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/- 1."""
    t = [1]
    while len(t) < n:
        t += [-x for x in t]
    return np.array(t[:n])

def _hill_climb(f, deadline):
    """Perform best-improvement local search using single-bit flips."""
    n = len(f)
    f = f.copy()
    current_score = _max_abs_autocorr(f)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_score = current_score
        best_idx = -1
        
        for i in range(n):
            if time.time() > deadline:
                break
            f[i] *= -1
            new_score = _max_abs_autocorr(f)
            f[i] *= -1
            
            if new_score < best_score:
                best_score = new_score
                best_idx = i
        
        if best_idx != -1:
            f[best_idx] *= -1
            current_score = best_score
            improved = True
            
    return f, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.RandomState(42)
    
    best_f = None
    best_score = float('inf')
    
    candidates = []
    candidates.append(_thue_morse(N))
    
    for _ in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    candidates.append(np.tile([-1.0, 1.0], N // 2))

    for f_init in candidates:
        if time.time() > deadline:
            break
        
        f_local, score_local = _hill_climb(f_init, deadline)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local
    
    if best_f is not None and time.time() < deadline - 10.0:
        f = best_f.copy()
        current_score = best_score
        n = len(f)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_score = current_score
            best_i = -1
            best_j = -1
            
            for i in range(n):
                if time.time() > deadline:
                    break
                f[i] *= -1
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    f[j] *= -1
                    new_score = _max_abs_autocorr(f)
                    f[j] *= -1
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_i = i
                        best_j = j
                f[i] *= -1
                
            if best_i != -1:
                f[best_i] *= -1
                f[best_j] *= -1
                current_score = best_score
                improved = True
                
        if current_score < best_score:
            best_score = current_score
            best_f = f

    return best_f
# EVOLVE-BLOCK-END
