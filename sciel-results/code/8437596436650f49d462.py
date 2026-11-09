import numpy as np
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 148.0
    N = 24
    k_arr = np.arange(N)
    
    # Precompute roots of unity for fast evaluation
    FFT_SIZE = 16384
    theta_grid = 2 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE
    roots = np.exp(1j * theta_grid)
    
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
    
    def eval_sup_fast(c):
        """Evaluate max |P(e^{i theta})| using FFT-based dense sampling."""
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        return float(np.max(np.abs(F)))
    
    def eval_at_theta(c, theta):
        """Evaluate P at a specific theta."""
        return float(np.abs(np.dot(c, np.exp(1j * theta * k_arr))))
    
    def golden_refine(c, theta_lo, theta_hi, n_iter=40):
        """Golden section search for max |P(e^{i theta})| in [theta_lo, theta_hi]."""
        inv_phi = (1 + 5**0.5) / 2
        phi = 1.0 / inv_phi
        a, b = theta_lo, theta_hi
        c1 = b - phi * (b - a)
        c2 = a + phi * (b - a)
        
        f1 = eval_at_theta(c, c1)
        f2 = eval_at_theta(c, c2)
        for _ in range(n_iter):
            if f1 < f2:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + phi * (b - a)
                f2 = eval_at_theta(c, c2)
            else:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - phi * (b - a)
                f1 = eval_at_theta(c, c1)
        return max(f1, f2)
    
    def find_all_peaks(c, n_peaks=5):
        """Find the top n_peaks local maxima of |P(e^{i theta})|."""
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        abs_F = np.abs(F)
        
        # Find top peaks in the FFT grid
        indices = np.argsort(abs_F)[::-1][:n_peaks * 3]  # Get more than needed
        
        # Cluster nearby peaks
        peaks = []
        used = set()
        for idx in indices:
            if idx in used:
                continue
            # Check if this peak is close to an already found peak
            is_new = True
            for p in peaks:
                if abs(idx - p) < FFT_SIZE // 100:  # Within ~3.6 degrees
                    is_new = False
                    break
            if is_new:
                peaks.append(idx)
                for j in range(max(0, idx - 5), min(FFT_SIZE, idx + 6)):
                    used.add(j)
            if len(peaks) >= n_peaks:
                break
        
        # Refine each peak with golden section
        dtheta = 2 * np.pi / FFT_SIZE
        refined_peaks = []
        for idx in peaks:
            theta = 2 * np.pi * idx / FFT_SIZE
            lo = max(0, theta - dtheta)
            hi = min(2 * np.pi, theta + dtheta)
            val = golden_refine(c, lo, hi, 35)
            refined_peaks.append(val)
        
        return max(refined_peaks) if refined_peaks else 0.0
    
    def find_peak_precise(c):
        """Find the true sup-norm peak using dense FFT grid + golden refinement."""
        return find_all_peaks(c, n_peaks=3)
    
    # Generate starting candidates
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
    
    # All double-bit flips of GRS
    for i in range(N):
        for j in range(i + 1, N):
            c = g.copy()
            c[i] = -c[i]
            c[j] = -c[j]
            candidates.append(c)
    
    # All double-bit flips of reversed GRS
    for i in range(N):
        for j in range(i + 1, N):
            c = gr.copy()
            c[i] = -c[i]
            c[j] = -c[j]
            candidates.append(c)
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(50):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates, keep best ones
    scored = []
    for c in candidates:
        if time.time() > deadline - 40.0:
            break
        val = eval_sup_fast(c)
        scored.append((val, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top candidates for local search
    top_k = min(50, len(scored))
    best_c = None
    best_val = float('inf')
    
    for val, c in scored[:top_k]:
        if time.time() > deadline - 30.0:
            break
        pv = find_peak_precise(c)
        if pv < best_val:
            best_val = pv
            best_c = c.copy()
    
    if best_c is None:
        best_c = g.copy()
        best_val = find_peak_precise(best_c)
    
    # Local search: single-bit flips with multiple restarts
    improved = True
    passes = 0
    while improved and passes < 100 and time.time() < deadline - 15.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        
        for j in range(N):
            if time.time() > deadline - 15.0:
                break
            trial = best_c.copy()
            trial[j] = -trial[j]
            quick = eval_sup_fast(trial)
            if quick < cur * 1.02:
                pv = find_peak_precise(trial)
                if pv < cur - 1e-10:
                    best_c = trial
                    cur = pv
                    improved = True
    
    # Double-bit flips
    improved = True
    passes = 0
    while improved and passes < 20 and time.time() < deadline - 8.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        
        for j in range(N):
            for l in range(j + 1, N):
                if time.time() > deadline - 8.0:
                    break
                trial = best_c.copy()
                trial[j] = -trial[j]
                trial[l] = -trial[l]
                quick = eval_sup_fast(trial)
                if quick < cur * 1.02:
                    pv = find_peak_precise(trial)
                    if pv < cur - 1e-10:
                        best_c = trial
                        cur = pv
                        improved = True
    
    # Triple-bit flips
    improved = True
    passes = 0
    while improved and passes < 5 and time.time() < deadline - 3.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        
        for j in range(N):
            for l in range(j + 1, N):
                for m in range(l + 1, N):
                    if time.time() > deadline - 3.0:
                        break
                    trial = best_c.copy()
                    trial[j] = -trial[j]
                    trial[l] = -trial[l]
                    trial[m] = -trial[m]
                    quick = eval_sup_fast(trial)
                    if quick < cur * 1.03:
                        pv = find_peak_precise(trial)
                        if pv < cur - 1e-10:
                            best_c = trial
                            cur = pv
                            improved = True
    
    out = [1 if v >= 0 else -1 for v in best_c]
    return out
