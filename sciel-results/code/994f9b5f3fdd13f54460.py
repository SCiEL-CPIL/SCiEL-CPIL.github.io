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
            num_2bit_trials = 10
            for _ in range(num_2bit_trials):
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
                    c[j] = -c[j]
                    
        if best_c_trial is not None:
            c = best_c_trial
            current_fast_peak = best_flip_peak
            improved = True
        else:
            # No improvement found, break
            break
            
    # Final accurate evaluation
    final_peak = _peak(c)
    return c, final_peak

def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s kill
    n = 24
    rng = np.random.default_rng(12345)
    candidates = []

    # Structured / known-good patterns
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
        p, q = p + q, p + [-x for x in q]
    candidates.append(np.array(p[:n], dtype=float))

    # Diverse random restarts
    for seed in range(60):
        r = np.random.default_rng(seed)
        candidates.append(r.choice([-1.0, 1.0], size=n))

    # Phase 1: fast screening, keep top candidates
    screened = []
    for c in candidates:
        if time.time() > deadline - 60:
            break
        pk = _peak_fast(c)
        screened.append((pk, c))
    screened.sort(key=lambda x: x[0])
    top = [c for _, c in screened[:24]]

    best_c = top[0]
    best_score = _peak(best_c)

    # Phase 2: local search on each top candidate
    for c in top:
        if time.time() > deadline - 12:
            break
        c_opt, peak_opt = _local_search(c, deadline - 8, rng)
        if peak_opt < best_score:
            best_score = peak_opt
            best_c = c_opt
        if best_score < 6.162:  # near the known optimum ~6.162
            break

    if best_c is None:
        best_c = candidates[0]
    return [int(x) for x in best_c]
