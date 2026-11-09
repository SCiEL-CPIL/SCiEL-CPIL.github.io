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
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        abs_F = np.abs(F)
        peak_idx = int(np.argmax(abs_F))
        
        dtheta = 2 * np.pi / FFT_SIZE
        window = 5
        best_local = 0
        best_theta = 2 * np.pi * peak_idx / FFT_SIZE
        for di in range(-window, window + 1):
            idx = (peak_idx + di) % FFT_SIZE
            if abs_F[idx] > best_local:
                best_local = abs_F[idx]
                best_theta = 2 * np.pi * idx / FFT_SIZE
        
        lo = max(0, best_theta - dtheta)
        hi = min(2 * np.pi, best_theta + dtheta)
        return golden_refine(c, lo, hi, 35)
    
    def local_search(start_c, time_limit):
        """Perform local search starting from start_c until time_limit."""
        best_c = start_c.copy()
        cur_val = find_peak_precise(best_c)
        improved = True
        passes = 0
        # Single bit flips
        while improved and passes < 100 and time.time() < time_limit:
            improved = False
            passes += 1
            for j in range(N):
                if time.time() > time_limit:
                    break
                trial = best_c.copy()
                trial[j] = -trial[j]
                quick = eval_sup(trial)
                # Use a tight filter
                if quick < cur_val * 1.005:
                    pv = find_peak_precise(trial)
                    if pv < cur_val - 1e-12:
                        best_c = trial
                        cur_val = pv
                        improved = True
        # Double bit flips
        improved = True
        passes = 0
        while improved and passes < 20 and time.time() < time_limit:
            improved = False
            passes += 1
            for j in range(N):
                for l in range(j + 1, N):
                    if time.time() > time_limit:
                        break
                    trial = best_c.copy()
                    trial[j] = -trial[j]
                    trial[l] = -trial[l]
                    quick = eval_sup(trial)
                    if quick < cur_val * 1.005:
                        pv = find_peak_precise(trial)
                        if pv < cur_val - 1e-12:
                            best_c = trial
                            cur_val = pv
                            improved = True
        return best_c, cur_val

    # Generate GRS
    g = grs(N)
    gr = g[::-1].copy()
    
    # Strategy: Run local search from multiple promising starts
    # 1. GRS
    # 2. Reversed GRS
    # 3. A few random starts
    
    best_overall_c = None
    best_overall_val = float('inf')
    
    starts = [g, gr]
    rng = np.random.RandomState(42)
    for _ in range(5):
        starts.append(rng.choice([-1.0, 1.0], size=N))
    
    # Allocate time budget
    # We have ~145s.
    # Each local search might take significant time.
    # Let's give each start 20s.
    
    for s in starts:
        if time.time() > deadline - 10.0:
            break
        t_limit = min(deadline - 10.0, time.time() + 20.0)
        c_res, v_res = local_search(s, t_limit)
        if v_res < best_overall_val:
            best_overall_val = v_res
            best_overall_c = c_res
    
    if best_overall_c is None:
        best_overall_c = g
        best_overall_val = find_peak_precise(best_overall_c)
        
    out = [1 if v >= 0 else -1 for v in best_overall_c]
    return out
