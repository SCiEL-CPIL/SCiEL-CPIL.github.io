import numpy as np
import time

def _eval_sup_norm(coeffs, N=8192):
    """Evaluate sup norm of polynomial on unit circle with golden-section refinement."""
    n = len(coeffs)
    coeffs = np.array(coeffs, dtype=np.float64)
    
    # Dense sampling
    theta = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * theta)
    
    # Evaluate polynomial using IFFT
    c_padded = np.zeros(N, dtype=np.complex128)
    c_padded[:n] = coeffs
    P = np.fft.ifft(c_padded) * N
    mags = np.abs(P)
    
    # Find peaks and refine with golden section
    top_k = min(30, N)
    peak_idx = np.argsort(mags)[-top_k:][::-1]
    
    best_mag = 0.0
    for idx in peak_idx:
        t0 = theta[idx]
        delta = 2 * np.pi / N
        a, b = t0 - delta, t0 + delta
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        
        # Precompute powers for speed
        k = np.arange(n)
        
        def f(t):
            zt = np.exp(1j * t)
            val = np.dot(coeffs, zt ** k)
            return np.abs(val)
        
        fc = f(c_pt)
        fd = f(d_pt)
        
        for _ in range(40):
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = f(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = f(d_pt)
        
        refined_val = max(fc, fd)
        if refined_val > best_mag:
            best_mag = refined_val
    
    return best_mag

def _fast_eval(coeffs, N=4096):
    """Faster evaluation without refinement, for local search."""
    n = len(coeffs)
    coeffs = np.array(coeffs, dtype=np.float64)
    theta = 2 * np.pi * np.arange(N) / N
    c_padded = np.zeros(N, dtype=np.complex128)
    c_padded[:n] = coeffs
    P = np.fft.ifft(c_padded) * N
    return np.max(np.abs(P))

def _spectral_init(n=24):
    """Initialize using spectral synthesis: inverse FFT of a flattened spectrum."""
    # Try various target spectra
    best_coeffs = None
    best_score = np.inf
    
    N = 1024
    k = np.arange(N)
    
    # Target 1: Flat (all ones)
    for taper in [1.0, 0.5, 0.25]:
        # Cosine-tapered spectrum
        target = np.ones(N)
        # Apply taper to reduce edge effects
        taper_mask = np.ones(N)
        w = int(N * taper)
        if w > 0:
            taper_mask[:w] = np.cos(np.linspace(np.pi/2, 0, w))
            taper_mask[-w:] = np.cos(np.linspace(0, np.pi/2, w))
        target = target * taper_mask
        
        # Inverse FFT to get time-domain coefficients
        coeffs_complex = np.fft.ifft(target)
        # Take real part and normalize to ±1
        coeffs_real = np.real(coeffs_complex[:n])
        coeffs_bin = np.sign(coeffs_real)
        # Ensure no zeros
        coeffs_bin[coeffs_bin == 0] = 1
        
        score = _fast_eval(coeffs_bin)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs_bin)
    
    # Target 2: Uniform with random phase
    rng = np.random.RandomState(42)
    for trial in range(50):
        phase = rng.uniform(0, 2*np.pi, N)
        target = np.exp(1j * phase)
        coeffs_complex = np.fft.ifft(target)
        coeffs_real = np.real(coeffs_complex[:n])
        coeffs_bin = np.sign(coeffs_real)
        coeffs_bin[coeffs_bin == 0] = 1
        
        score = _fast_eval(coeffs_bin)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs_bin)
    
    # Target 3: Golay-Rudin-Shapiro as baseline
    def grs_coeffs(m):
        p, q = [1], [1]
        while len(p) < m:
            p, q = p + q, p + [-c for c in q]
        return p[:m]
    
    grs = grs_coeffs(n)
    score = _fast_eval(grs)
    if score < best_score:
        best_score = score
        best_coeffs = list(grs)
    
    return best_coeffs

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n = 24
    
    # Phase 1: Spectral synthesis initialization
    best_coeffs = _spectral_init(n)
    best_norm = _eval_sup_norm(best_coeffs)
    
    # Phase 2: Vectorized local search
    # Precompute DFT basis for fast evaluation
    N = 4096
    theta = 2 * np.pi * np.arange(N) / N
    k = np.arange(n)
    # Basis: basis[i, j] = exp(i * i * theta_j)
    basis = np.exp(1j * np.outer(k, theta))  # shape (n, N)
    
    def _eval_with_basis(coeffs):
        """Evaluate sup norm using precomputed basis."""
        coeffs = np.array(coeffs, dtype=np.float64)
        P = basis.T @ coeffs  # shape (N,)
        return np.max(np.abs(P))
    
    # Local search: single flips
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            old_val = best_coeffs[i]
            best_coeffs[i] = -old_val
            new_norm = _eval_with_basis(best_coeffs)
            if new_norm < best_norm - 1e-12:
                best_norm = new_norm
                improved = True
            else:
                best_coeffs[i] = old_val
    
    # Local search: pair flips
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    break
                old_i, old_j = best_coeffs[i], best_coeffs[j]
                best_coeffs[i] = -old_i
                best_coeffs[j] = -old_j
                new_norm = _eval_with_basis(best_coeffs)
                if new_norm < best_norm - 1e-12:
                    best_norm = new_norm
                    improved = True
                else:
                    best_coeffs[i] = old_i
                    best_coeffs[j] = old_j
            if time.time() > deadline - 5:
                break
    
    # Phase 3: Simulated annealing for escaping local minima
    if time.time() < deadline - 20:
        current_coeffs = list(best_coeffs)
        current_norm = best_norm
        T = 0.01
        T_min = 0.0001
        cooling = 0.995
        
        for step in range(50000):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            # Random single flip
            i = np.random.randint(n)
            old_val = current_coeffs[i]
            current_coeffs[i] = -old_val
            new_norm = _eval_with_basis(current_coeffs)
            
            delta = new_norm - current_norm
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_norm = new_norm
                if new_norm < best_norm - 1e-12:
                    best_norm = new_norm
                    best_coeffs = list(current_coeffs)
            else:
                current_coeffs[i] = old_val
            
            T *= cooling
    
    # Phase 4: Final refinement with accurate evaluation
    if time.time() < deadline - 2:
        # Do a few more single flips with accurate evaluation
        for _ in range(10):
            if time.time() > deadline - 1:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 0.5:
                    break
                old_val = best_coeffs[i]
                best_coeffs[i] = -old_val
                new_norm = _eval_sup_norm(best_coeffs)
                if new_norm < best_norm - 1e-15:
                    best_norm = new_norm
                    improved = True
                else:
                    best_coeffs[i] = old_val
            if not improved:
                break
    
    # Ensure all coefficients are ±1
    best_coeffs = [1 if c > 0 else -1 for c in best_coeffs]
    
    return best_coeffs
