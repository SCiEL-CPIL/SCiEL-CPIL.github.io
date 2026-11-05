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
    """Fast peak estimate via FFT (no refinement)."""
    n = len(c)
    N = 4096
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F = np.fft.ifft(cf) * N
    return np.abs(F).max()

def _peak(c):
    """Accurate peak: dense FFT sampling + golden-section refinement of top peaks."""
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

def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin
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
    for seed in range(30):
        r = np.random.default_rng(seed)
        candidates.append(r.choice([-1.0, 1.0], size=n))
        
    best_c = None
    best_score = float('inf')
    
    for c_init in candidates:
        if time.time() > deadline - 5:
            break
            
        c = c_init.copy()
        
        # Iterated Local Search
        while time.time() < deadline - 2:
            # 1. Local Search to local minimum
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                best_new_peak = _peak_fast(c)
                best_flip_idx = -1
                
                # 1-bit flips
                for i in range(n):
                    c[i] = -c[i]
                    p_trial = _peak_fast(c)
                    if p_trial < best_new_peak:
                        best_new_peak = p_trial
                        best_flip_idx = i
                    c[i] = -c[i]
                    
                if best_flip_idx != -1:
                    c[best_flip_idx] = -c[best_flip_idx]
                    improved = True
                    continue
                
                # 2-bit flips (limited attempts)
                found_2bit = False
                for _ in range(10):
                    if time.time() > deadline - 2:
                        break
                    i, j = rng.choice(n, size=2, replace=False)
                    c[i] = -c[i]
                    c[j] = -c[j]
                    p_trial = _peak_fast(c)
                    if p_trial < best_new_peak:
                        best_new_peak = p_trial
                        found_2bit = True
                        break
                    else:
                        c[i] = -c[i]
                        c[j] = -c[j]
                        
                if found_2bit:
                    improved = True
                else:
                    # Stuck in local minimum
                    break
            
            # 2. Evaluate exact peak of this local minimum
            current_peak = _peak(c)
            if current_peak < best_score:
                best_score = current_peak
                best_c = c.copy()
                if best_score < 6.162:
                    return [int(x) for x in best_c]
            
            # 3. Perturb to escape
            # Flip 4 random bits
            flip_idx = rng.choice(n, size=4, replace=False)
            c[flip_idx] = -c[flip_idx]
            
    if best_c is None:
        best_c = candidates[0]
        
    return [int(x) for x in best_c]
