import numpy as np
import time
import random

def _grs_sequence(n):
    """Generate the Golay-Rudin-Shapiro sequence of length n."""
    a = [0] * n
    a[0] = 1
    for i in range(1, n):
        k = i // 2
        if i % 2 == 0:
            a[i] = a[k]
        else:
            a[i] = -a[k]
    return np.array(a, dtype=np.float64)

def _peak_val(c, thetas):
    """Evaluate max |P(z)| on given thetas."""
    k = np.arange(len(c))
    Z = np.exp(1j * thetas[:, None] * k[None, :])
    return float(np.max(np.abs(Z @ c)))

def _parabolic_refine(c, theta_lo, theta_hi, n_iter=10):
    """Refine the peak location using parabolic interpolation."""
    a, b = theta_lo, theta_hi
    # Initial points
    c1 = a + (b - a) / 3.0
    c2 = b - (b - a) / 3.0
    f1 = _peak_val(c, np.array([c1]))
    f2 = _peak_val(c, np.array([c2]))
    
    for _ in range(n_iter):
        # Parabolic interpolation
        # f(x) = A*x^2 + B*x + C
        # Vertex at x = -B/(2A)
        # Using three points: (a, fa), (c1, f1), (b, fb) -- wait, I need 3 points.
        # Let's use (a, fa), (c1, f1), (c2, f2) ? No, a and b are the bounds.
        # Let's evaluate at a and b? That's expensive.
        # Standard parabolic interpolation uses 3 points.
        # Let's maintain 3 points: x0, x1, x2 with f0, f1, f2.
        # Start with x0=a, x1=c1, x2=c2.
        pass
    
    # Simpler: Golden section is robust. Parabolic is faster if it works.
    # Let's stick to a robust fast refinement.
    # The previous golden section took 20 iterations.
    # Let's try 10 iterations of golden section, it's fast enough.
    return _golden_refine_peak(c, a, b, n_iter=10)

def _golden_refine_peak(c, theta_lo, theta_hi, n_iter=10):
    """Refine the peak location near a given interval."""
    inv_phi = (1 + 5**0.5) / 2
    phi = 1.0 / inv_phi
    a, b = theta_lo, theta_hi
    c1 = b - phi * (b - a)
    c2 = a + phi * (b - a)
    f1 = _peak_val(c, np.array([c1]))
    f2 = _peak_val(c, np.array([c2]))
    for _ in range(n_iter):
        if f1 < f2:
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - phi * (b - a)
            f1 = _peak_val(c, np.array([c1]))
        else:
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + phi * (b - a)
            f2 = _peak_val(c, np.array([c2]))
    return max(f1, f2)

def _find_global_peak(c, M_coarse=1024):
    """Find the global peak by coarse scan + golden refinement."""
    thetas_coarse = np.linspace(0, 2*np.pi, M_coarse, endpoint=False)
    k = np.arange(len(c))
    Z = np.exp(1j * thetas_coarse[:, None] * k[None, :])
    vals = np.abs(Z @ c)
    peak_idx = int(np.argmax(vals))
    dtheta = 2*np.pi / M_coarse
    lo = peak_idx * dtheta
    hi = (peak_idx + 1) * dtheta
    return _golden_refine_peak(c, lo, hi)

def construct():
    start = time.time()
    deadline = start + 115.0
    N = 24
    
    # Precompute FFT basis for fast evaluation
    M_fft = 256  # Power of 2, > 24
    # We need to evaluate P(z) at z = exp(i * 2*pi * j / M_fft)
    # This is the DFT of the coefficients.
    # P(z) = sum c_k z^k.
    # If we zero-pad c to length M_fft, then FFT(c) gives P(exp(-i 2 pi j / M_fft)).
    # We want P(exp(i 2 pi j / M_fft)).
    # Note that P(exp(i theta)) = conj(P(exp(-i theta))) if c is real.
    # So |P(exp(i theta))| = |P(exp(-i theta))|.
    # So we can just take the absolute value of the FFT of the zero-padded coefficients.
    
    # Let's verify.
    # c = [1, -1, 1, -1, ...]
    # P(z) = 1 - z + z^2 - z^3 ...
    # FFT(c) at index j is sum c_k exp(-i 2 pi j k / M).
    # This is P(exp(-i 2 pi j / M)).
    # |P(exp(-i theta))| = |1 - exp(-i theta) + ...|
    # |P(exp(i theta))| = |1 - exp(i theta) + ...|
    # Since c is real, P(exp(i theta)) = conj(P(exp(-i theta))).
    # So the magnitudes are identical.
    # So we can use np.fft.fft(c_padded) and take abs.
    
    c_padded = np.zeros(M_fft, dtype=np.float64)
    
    # Initialize with GRS
    c = _grs_sequence(N)
    c_padded[:N] = c
    
    # Initial peak
    cur_peak = _find_global_peak(c)
    best_c = c.copy()
    best_peak = cur_peak
    
    # Local search
    improved = True
    passes = 0
    while improved and passes < 50 and time.time() < deadline - 5.0:
        improved = False
        passes += 1
        
        # Calculate current FFT
        cur_fft = np.fft.fft(c_padded)
        cur_mags = np.abs(cur_fft)
        cur_coarse_peak = float(np.max(cur_mags))
        
        # Try single-bit flips
        best_trial_c = None
        best_trial_coarse_peak = float('inf')
        
        for j in range(N):
            # Flip bit j
            # c_new[j] = -c[j]
            # delta = -2 * c[j] * z^j
            # In FFT domain: delta_fft = -2 * c[j] * exp(-i 2 pi j k / M)
            # This is a complex vector.
            # We can precompute the FFT of a single impulse at j?
            # FFT of delta_c (where delta_c[j] = -2*c[j], others 0)
            # delta_fft[k] = -2*c[j] * exp(-i 2 pi j k / M)
            
            # To speed this up, we can precompute the basis vectors for each j.
            # But N=24, M=256. 24 * 256 complex numbers is small.
            
            # Let's just compute the new FFT directly?
            # np.fft.fft is fast.
            # But we can do incremental update.
            
            # Let's precompute the "flip basis" vectors.
            # flip_basis[j] = FFT of vector with -2 at j, 0 elsewhere.
            # Actually, the change in c is -2*c[j] at index j.
            # So delta_c has -2*c[j] at j.
            # delta_fft = -2*c[j] * exp(-i 2 pi j * np.arange(M_fft) / M_fft)
            
            # We can precompute W[j] = exp(-i 2 pi j * np.arange(M_fft) / M_fft)
            
            if not hasattr(construct, '_flip_bases'):
                construct._flip_bases = []
                for j in range(N):
                    w = np.exp(-1j * 2 * np.pi * j * np.arange(M_fft) / M_fft)
                    construct._flip_bases.append(w)
            
            w = construct._flip_bases[j]
            delta_fft = -2.0 * c[j] * w
            new_fft = cur_fft + delta_fft
            new_mags = np.abs(new_fft)
            new_coarse_peak = float(np.max(new_mags))
            
            if new_coarse_peak < best_trial_coarse_peak:
                best_trial_coarse_peak = new_coarse_peak
                best_trial_c = c.copy()
                best_trial_c[j] = -best_trial_c[j]
        
        if best_trial_coarse_peak < cur_coarse_peak - 1e-6:
            # Refine the best trial
            trial_c_padded = np.zeros(M_fft, dtype=np.float64)
            trial_c_padded[:N] = best_trial_c
            trial_peak = _find_global_peak(best_trial_c)
            if trial_peak < best_peak - 1e-12:
                c = best_trial_c
                best_c = c.copy()
                best_peak = trial_peak
                improved = True
        
        # Try double-bit flips if time permits and single didn't improve
        if not improved and time.time() < deadline - 20.0:
            best_trial_c = None
            best_trial_coarse_peak = float('inf')
            
            for j in range(N):
                for l in range(j+1, N):
                    # Flip j and l
                    w_j = construct._flip_bases[j]
                    w_l = construct._flip_bases[l]
                    delta_fft = -2.0 * c[j] * w_j - 2.0 * c[l] * w_l
                    new_fft = cur_fft + delta_fft
                    new_mags = np.abs(new_fft)
                    new_coarse_peak = float(np.max(new_mags))
                    
                    if new_coarse_peak < best_trial_coarse_peak:
                        best_trial_coarse_peak = new_coarse_peak
                        best_trial_c = c.copy()
                        best_trial_c[j] = -best_trial_c[j]
                        best_trial_c[l] = -best_trial_c[l]
            
            if best_trial_coarse_peak < cur_coarse_peak - 1e-6:
                trial_peak = _find_global_peak(best_trial_c)
                if trial_peak < best_peak - 1e-12:
                    c = best_trial_c
                    best_c = c.copy()
                    best_peak = trial_peak
                    improved = True

    out = [1 if v >= 0 else -1 for v in best_c]
    return out
