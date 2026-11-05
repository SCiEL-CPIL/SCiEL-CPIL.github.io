import numpy as np
import time

def _eval_at(c, t):
    """Evaluate |P(exp(2j*pi*t))| using Horner's method."""
    z = np.exp(2j * np.pi * t)
    acc = 0j
    for coeff in reversed(c):
        acc = acc * z + coeff
    return np.abs(acc)

def _peak_fast(c):
    """
    Fast peak estimate via FFT (no refinement). 
    Used for screening in local search.
    """
    n = len(c)
    N = 4096
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F = np.fft.ifft(cf) * N
    return np.abs(F).max()

def _peak(c):
    """
    Accurate peak: dense FFT sampling + golden-section refinement of top peaks.
    Used for final evaluation and initial candidate ranking.
    """
    n = len(c)
    N = 8192
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F = np.fft.ifft(cf) * N
    mags = np.abs(F)
    
    top_k = 8
    idx = np.argpartition(mags, -top_k)[-top_k:]
    best_peak = float(mags[idx].max())
    
    gr = (np.sqrt(5) - 1) / 2
    for k in idx:
        lo = max(0.0, (k - 5) / N)
        hi = min(1.0, (k + 6) / N)
        a, b = lo, hi
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = _eval_at(c, c1)
        f2 = _eval_at(c, c2)
        
        for _ in range(35):
            if f1 < f2:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = _eval_at(c, c2)
            else:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = _eval_at(c, c1)
        
        v = max(f1, f2)
        if v > best_peak:
            best_peak = v
            
    return best_peak

def _local_search(c, deadline, rng):
    """
    Local search with 1-bit and 2-bit flips.
    Uses _peak_fast for trial evaluations to maximize iterations.
    """
    c = c.copy()
    current_fast_peak = _peak_fast(c)
    n = len(c)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        best_flip_peak = current_fast_peak
        best_c_trial = None
        
        # 1-bit flips
        for i in range(n):
            if time.time() > deadline:
                break
            c[i] = -c[i]
            p_trial = _peak_fast(c)
            if p_trial < best_flip_peak:
                best_flip_peak = p_trial
                best_c_trial = c.copy()
            else:
                c[i] = -c[i] # revert
        
        # 2-bit flips (to escape local minima)
        if time.time() < deadline:
            # Try a few random 2-bit flips
            for _ in range(10):
                if time.time() > deadline:
                    break
                i, j = rng.choice(n, size=2, replace=False)
                c[i] = -c[i]
                c[j] = -c[j]
                p_trial = _peak_fast(c)
                if p_trial < best_flip_peak:
                    best_flip_peak = p_trial
                    best_c_trial = c.copy()
                else:
                    c[i] = -c[i]
                    c[j] = -c[j] # revert

        if best_c_trial is not None:
            c = best_c_trial
            current_fast_peak = best_flip_peak
            improved = True
        else:
            break
            
    # Return the accurate peak of the best found local minimum
    final_accurate_peak = _peak(c)
    return c, final_accurate_peak

def _spectral_synthesis(rng, n=24, num_samples=100):
    """
    Generate candidates via spectral synthesis: inverse FFT of target waveforms.
    Creates globally coherent starting points by leveraging time-frequency duality.
    """
    candidates = []
    
    for _ in range(num_samples):
        # Generate a random smooth spectrum
        N = 1024
        # Create a target waveform: combination of flat and cosine-tapered
        if rng.random() < 0.5:
            # Flat spectrum with random phase
            target = np.ones(N, dtype=complex) * np.exp(2j * np.pi * rng.random(N))
        else:
            # Cosine-tapered spectrum
            t = np.linspace(-1, 1, N)
            taper = 0.5 + 0.5 * np.cos(np.pi * t)
            target = taper * np.exp(2j * np.pi * rng.random(N))
        
        # Add some randomness to the amplitude
        target *= (0.5 + 0.5 * rng.random(N))
        
        # Inverse FFT to get time-domain signal
        time_signal = np.fft.ifft(target) * N
        time_signal = np.real(time_signal[:n])
        
        # Threshold to ±1
        c = np.sign(time_signal)
        # Handle zeros
        c[c == 0] = 1.0
        
        candidates.append(c)
    
    return candidates

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin
    
    n = 24
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # Known optimal patterns / structured candidates
    candidates.append(np.array([1]*n, dtype=float))
    candidates.append(np.array([1,-1]*12, dtype=float))
    candidates.append(np.array([1,1,-1,-1]*6, dtype=float))
    candidates.append(np.array([1,-1,-1]*8, dtype=float))
    candidates.append(np.array([1,1,1,-1,-1,-1]*4, dtype=float))
    candidates.append(np.array([1,-1,1,1,-1,1,1,-1]*3, dtype=float))
    candidates.append(np.array([1,1,1,1,-1,-1,-1,-1]*3, dtype=float))
    candidates.append(np.array([1,-1,-1,-1,1,1,1,1]*3, dtype=float))
    
    # GRS construction
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:n], dtype=float))
    
    # Spectral synthesis candidates (key improvement)
    spectral_candidates = _spectral_synthesis(rng, n, num_samples=50)
    candidates.extend(spectral_candidates)
    
    # Random restarts
    for _ in range(30):
        candidates.append(rng.choice([-1.0, 1.0], size=n))
        
    # Phase 1: Fast screening to find promising candidates
    # We use _peak_fast here to sort candidates quickly
    screened = []
    for c in candidates:
        if time.time() > deadline - 60:
            break
        pk = _peak_fast(c)
        screened.append((pk, c))
        
    screened.sort(key=lambda x: x[0])
    
    # Select top candidates for detailed local search
    # We want to explore a diverse set of good starting points
    top_candidates = [c for _, c in screened[:30]]
    
    best_c = None
    best_score = float('inf')
    
    # Phase 2: Local search on top candidates
    for c_start in top_candidates:
        if time.time() > deadline - 10:
            break
            
        # Run local search with a sub-deadline to allow multiple attempts
        # The local search function handles its own timing, but we give it a deadline
        # We allocate time based on remaining budget
        sub_deadline = deadline - 5.0
        
        c_opt, peak_opt = _local_search(c_start, sub_deadline, rng)
        
        if peak_opt < best_score:
            best_score = peak_opt
            best_c = c_opt
            
        # If we find a very good score, we can stop early or be more aggressive
        # The exhaustive optimum is ~6.162 (score ~1.258)
        # If we get close, we might spend more time polishing
        if best_score < 6.17:
             # Continue trying to improve, but local search usually converges fast
             pass

    if best_c is None:
        # Fallback to best from screening
        best_c = top_candidates[0] if top_candidates else candidates[0]
        best_score = _peak(best_c)
        
    return [int(x) for x in best_c]
