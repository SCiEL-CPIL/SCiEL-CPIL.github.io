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
    Fast peak estimation using FFT with moderate resolution.
    Returns approximate max |P(z)| on unit circle.
    """
    n = len(c)
    N = 2048
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F = np.fft.ifft(cf) * N
    mags = np.abs(F)
    return mags.max()

def _peak_refined(c):
    """
    Accurate peak estimation: FFT + golden-section refinement of top peaks.
    """
    n = len(c)
    N = 4096
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F = np.fft.ifft(cf) * N
    mags = np.abs(F)
    
    top_k = 8
    idx = np.argpartition(mags, -top_k)[-top_k:]
    best_peak = mags[idx].max()
    
    gr = (np.sqrt(5) - 1) / 2
    for k in idx:
        lo = max(0.0, (k - 4) / N)
        hi = min(1.0, (k + 5) / N)
        a, b = lo, hi
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = _eval_at(c, c1)
        f2 = _eval_at(c, c2)
        
        for _ in range(40):
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

def _local_search(c, deadline):
    """
    Local search using fast peak evaluation for candidate selection,
    then refined evaluation for final acceptance.
    """
    c = c.copy()
    current_peak = _peak_fast(c)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        best_flip = -1
        best_flip_peak = current_peak
        best_c_trial = None
        
        for i in range(len(c)):
            if time.time() > deadline:
                break
            c[i] = -c[i]
            p_trial = _peak_fast(c)
            if p_trial < best_flip_peak:
                best_flip_peak = p_trial
                best_flip = i
                best_c_trial = c.copy()
            else:
                c[i] = -c[i]  # revert
                
        if best_flip >= 0:
            c = best_c_trial
            current_peak = best_flip_peak
            improved = True
        else:
            break
            
    # Final refined evaluation
    final_peak = _peak_refined(c)
    return c, final_peak

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin
    
    n = 24
    
    # Generate diverse initial candidates
    candidates = []
    
    # GRS construction
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:n], dtype=float))
    
    # Random restarts with different seeds
    for seed in range(80):
        rng = np.random.default_rng(seed)
        candidates.append(rng.choice([-1.0, 1.0], size=n))
        
    # Structured patterns
    candidates.append(np.array([1, -1] * 12, dtype=float))
    candidates.append(np.array([1, 1, -1, -1] * 6, dtype=float))
    candidates.append(np.array([1, -1, -1] * 8, dtype=float))
    candidates.append(np.array([1, 1, 1, -1, -1, -1] * 4, dtype=float))
    candidates.append(np.array([1, -1, 1, 1, -1, -1, 1, -1] * 3, dtype=float))

    best_c = None
    best_score = float('inf')
    
    # Phase 1: Quick screening with fast peak
    screened = []
    for c in candidates:
        if time.time() > deadline - 60:
            break
        pk = _peak_fast(c)
        screened.append((pk, c))
    
    # Sort by fast peak, take top 20
    screened.sort(key=lambda x: x[0])
    top_candidates = [c for _, c in screened[:20]]
    
    # Phase 2: Local search on top candidates
    for c in top_candidates:
        if time.time() > deadline - 10:
            break
        
        c_opt, peak_opt = _local_search(c, deadline - 5)
        
        if peak_opt < best_score:
            best_score = peak_opt
            best_c = c_opt
            
        # Early termination if we hit near-optimal
        if best_score < 6.162:
            break

    if best_c is None:
        best_c = candidates[0]
        
    return [int(x) for x in best_c]
