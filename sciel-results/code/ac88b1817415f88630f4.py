import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 145.0
    N = 24
    k_arr = np.arange(N)
    
    # Precompute FFT-based evaluation on dense grid
    # Use a large FFT size for accurate sup-norm estimation
    FFT_SIZE = 8192
    # We evaluate P(z) at z = exp(2*pi*i*m/FFT_SIZE) for m=0..FFT_SIZE-1
    # This gives us |P| on a fine grid
    
    def grs(n):
        a = np.zeros(n)
        a[0] = 1.0
        for i in range(1, n):
            kk = i >> 1
            if i & 1:
                a[i] = -a[kk]
            else:
                a[i] = a[kk]
        return a
    
    def eval_sup(c, fft_size=FFT_SIZE):
        """Evaluate max |P(e^{i theta})| using FFT-based dense sampling."""
        # c is a numpy array of length N with values in {+1, -1}
        # We want max over theta of |sum c_k e^{i k theta}|
        # Using FFT: if we zero-pad c to length fft_size and take FFT,
        # we get P at the fft_size-th roots of unity.
        c_pad = np.zeros(fft_size, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        return float(np.max(np.abs(F)))
    
    def golden_refine(c, theta_lo, theta_hi, n_iter=30):
        """Golden section search for max |P(e^{i theta})| in [theta_lo, theta_hi]."""
        inv_phi = (1 + 5**0.5) / 2
        phi = 1.0 / inv_phi
        a, b = theta_lo, theta_hi
        c1 = b - phi * (b - a)
        c2 = a + phi * (b - a)
        
        def f(theta):
            return float(np.abs(np.sum(c * np.exp(1j * theta * k_arr))))
        
        f1 = f(c1)
        f2 = f(c2)
        for _ in range(n_iter):
            if f1 < f2:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + phi * (b - a)
                f2 = f(c2)
            else:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - phi * (b - a)
                f1 = f(c1)
        return max(f1, f2)
    
    def find_peak_precise(c):
        """Find the true sup-norm peak using dense FFT grid + golden refinement."""
        # First, find approximate peak location using FFT grid
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        abs_F = np.abs(F)
        peak_idx = int(np.argmax(abs_F))
        
        # The peak is near theta = 2*pi*peak_idx/FFT_SIZE
        dtheta = 2 * np.pi / FFT_SIZE
        # Search in a window around the peak
        window = 5  # look at 5 grid points on each side
        best_local = 0
        best_theta = 2 * np.pi * peak_idx / FFT_SIZE
        for di in range(-window, window + 1):
            idx = (peak_idx + di) % FFT_SIZE
            if abs_F[idx] > best_local:
                best_local = abs_F[idx]
                best_theta = 2 * np.pi * idx / FFT_SIZE
        
        # Golden section refine in [best_theta - dtheta, best_theta + dtheta]
        lo = max(0, best_theta - dtheta)
        hi = min(2 * np.pi, best_theta + dtheta)
        return golden_refine(c, lo, hi, 35)
    
    # Generate many starting candidates
    candidates = []
    
    # GRS and variants
    g = grs(N)
    candidates.append(g)
    candidates.append(g[::-1].copy())
    candidates.append(-g)
    candidates.append(-g[::-1].copy())
    
    # Cyclic shifts of GRS
    for shift in range(1, N):
        candidates.append(np.roll(g, shift))
        candidates.append(np.roll(g, shift)[::-1].copy())
    
    # All single-bit flips of GRS
    for i in range(N):
        c = g.copy()
        c[i] = -c[i]
        candidates.append(c)
    
    # All single-bit flips of reversed GRS
    gr = g[::-1].copy()
    for i in range(N):
        c = gr.copy()
        c[i] = -c[i]
        candidates.append(c)
    
    # Random starts
    rng = np.random.RandomState(12345)
    for _ in range(20):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates, keep best ones
    scored = []
    for c in candidates:
        if time.time() > deadline - 30.0:
            break
        # Quick eval with FFT
        val = eval_sup(c)
        scored.append((val, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top candidates for local search
    top_k = min(30, len(scored))
    best_c = None
    best_val = float('inf')
    
    for val, c in scored[:top_k]:
        if time.time() > deadline - 20.0:
            break
        # Precise evaluation
        pv = find_peak_precise(c)
        if pv < best_val:
            best_val = pv
            best_c = c.copy()
    
    if best_c is None:
        best_c = g.copy()
        best_val = find_peak_precise(best_c)
    
    # Local search: single-bit flips
    improved = True
    passes = 0
    while improved and passes < 50 and time.time() < deadline - 10.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        
        for j in range(N):
            if time.time() > deadline - 10.0:
                break
            trial = best_c.copy()
            trial[j] = -trial[j]
            # Quick filter
            quick = eval_sup(trial)
            if quick < cur * 1.01:  # within 1%
                pv = find_peak_precise(trial)
                if pv < cur - 1e-12:
                    best_c = trial
                    cur = pv
                    improved = True
    
    # Double-bit flips
    improved = True
    passes = 0
    while improved and passes < 10 and time.time() < deadline - 5.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        
        for j in range(N):
            for l in range(j + 1, N):
                if time.time() > deadline - 5.0:
                    break
                trial = best_c.copy()
                trial[j] = -trial[j]
                trial[l] = -trial[l]
                quick = eval_sup(trial)
                if quick < cur * 1.01:
                    pv = find_peak_precise(trial)
                    if pv < cur - 1e-12:
                        best_c = trial
                        cur = pv
                        improved = True
    
    out = [1 if v >= 0 else -1 for v in best_c]
    return out
