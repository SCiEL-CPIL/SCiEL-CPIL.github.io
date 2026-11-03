"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return np.max(np.abs(ac)) / (s * s)

def _score_from_H(H):
    """Compute score directly from frequency components H (length N, complex)."""
    h = np.fft.ifft(H).real
    return _score(h)

def _spectral_objective(x):
    """
    Objective in frequency domain using spectral decomposition.
    x encodes H[0] (real) and H[k] for k=1..N/2 (real, imag).
    We minimize max_k |H[k]|^2 / H[0]^2 which equals the max autoconv ratio
    for the circular convolution case. For linear, we use the actual score.
    """
    n_half = N // 2
    H = np.zeros(N, dtype=complex)
    H[0] = x[0]
    for k in range(1, n_half + 1):
        H[k] = x[2*k - 1] + 1j * x[2*k]
        H[N - k] = np.conj(H[k])
    
    H0 = H[0]
    if abs(H0) < 1e-12:
        return 1e10
    
    # The ratio for circular autoconv is max_k |H[k]|^2 / |H[0]|^2
    # But we need linear autoconv. Let's compute the actual score.
    h = np.fft.ifft(H).real
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_optimize():
    """
    Optimize in frequency domain using the structure of the problem.
    Key insight: The autoconvolution at lag k is (h * h)[k] = sum_j h[j] h[j+k].
    In frequency domain, the linear autoconvolution spectrum is |H_pad|^2.
    
    Strategy: Use the fact that we want to minimize max|autoconv|/(sum h)^2.
    If we normalize sum(h) = 1, we minimize max|autoconv|.
    
    For the circular case, max|autoconv_circ| = max_k |H[k]|^2 / N.
    So we want to minimize max_k |H[k]|^2 subject to |H[0]| = 1 (since sum h = H[0]/N * N = H[0]).
    
    Actually, sum(h) = H[0] where H is the DFT. So if we fix H[0] = 1, sum(h) = 1.
    Then we minimize max_k |H[k]|^2 for k != 0, which gives the circular ratio.
    For linear, it's slightly different but correlated.
    
    Let's use a direct approach: minimize the max of |H[k]|^2 / H[0]^2.
    """
    t_start = time.time()
    deadline = t_start + 80.0
    
    n_half = N // 2
    n_vars = 1 + 2 * n_half
    
    # Initializations
    initializations = []
    
    # 1. Flat spectrum (all H[k] equal)
    x_flat = np.zeros(n_vars)
    x_flat[0] = 1.0
    for k in range(1, n_half + 1):
        x_flat[2*k - 1] = 1.0 / N
        x_flat[2*k] = 0.0
    initializations.append(x_flat)
    
    # 2. Concentrated at DC (ideal but infeasible for real h with small support)
    x_dc = np.zeros(n_vars)
    x_dc[0] = 1.0
    initializations.append(x_dc)
    
    # 3. Cosine window in frequency
    x_cos = np.zeros(n_vars)
    x_cos[0] = 1.0
    for k in range(1, n_half + 1):
        x_cos[2*k - 1] = 0.5 * (1 + np.cos(np.pi * k / (n_half + 1)))
        x_cos[2*k] = 0.0
    initializations.append(x_cos)
    
    # 4. Gaussian decay in frequency
    x_gauss = np.zeros(n_vars)
    x_gauss[0] = 1.0
    sigma = n_half / 4.0
    for k in range(1, n_half + 1):
        x_gauss[2*k - 1] = np.exp(-k**2 / (2 * sigma**2))
        x_gauss[2*k] = 0.0
    initializations.append(x_gauss)
    
    # 5. Random perturbations of DC-concentrated
    rng = np.random.default_rng(12345)
    for scale in [0.001, 0.01, 0.05]:
        x_rand = np.zeros(n_vars)
        x_rand[0] = 1.0
        x_rand[1:] = rng.normal(0, scale, size=2*n_half)
        initializations.append(x_rand)
    
    # 6. Alternating signs in frequency to create oscillatory h
    x_alt = np.zeros(n_vars)
    x_alt[0] = 1.0
    for k in range(1, n_half + 1):
        x_alt[2*k - 1] = 0.01 * ((-1) ** k)
        x_alt[2*k] = 0.0
    initializations.append(x_alt)
    
    best_x = initializations[0].copy()
    best_score = _spectral_objective(best_x)
    
    for x_init in initializations:
        if time.time() > deadline - 15:
            break
        
        try:
            # Use L-BFGS-B with bounds to keep values reasonable
            bounds = [(None, None)] + [(-10, 10)] * (2 * n_half)
            result = minimize(
                _spectral_objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Local refinement: coordinate descent on frequency components
    if time.time() < deadline - 10:
        x_local = best_x.copy()
        best_local = best_score
        
        for iteration in range(5):
            improved = False
            for idx in range(1, n_vars):  # Skip DC
                if time.time() > deadline - 5:
                    break
                # Try small perturbations
                for delta in [0.0001, -0.0001, 0.001, -0.001, 0.01, -0.01]:
                    x_local[idx] += delta
                    new_val = _spectral_objective(x_local)
                    if new_val < best_local:
                        best_local = new_val
                        improved = True
                    else:
                        x_local[idx] -= delta
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_x = x_local.copy()
    
    # Convert to time domain
    H = np.zeros(N, dtype=complex)
    H[0] = best_x[0]
    for k in range(1, n_half + 1):
        H[k] = best_x[2*k - 1] + 1j * best_x[2*k]
        H[N - k] = np.conj(H[k])
    
    h = np.fft.ifft(H).real
    
    return h, best_score

def _time_domain_refine(h):
    """Local search in time domain with sign flips and small perturbations."""
    t_start = time.time()
    deadline = t_start + 20.0
    
    h_refined = h.copy()
    best_local = _score(h_refined)
    
    # Sign flips
    for _ in range(3):
        improved = False
        for idx in range(N):
            if time.time() > deadline - 5:
                break
            h_refined[idx] = -h_refined[idx]
            new_val = _score(h_refined)
            if new_val < best_local:
                best_local = new_val
                improved = True
            else:
                h_refined[idx] = -h_refined[idx]
        if not improved:
            break
    
    # Small additive perturbations
    for _ in range(2):
        improved = False
        rng = np.random.default_rng(int(time.time() * 1000) % 10000)
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            idx = rng.integers(0, N)
            delta = rng.normal(0, 0.005)
            h_refined[idx] += delta
            new_val = _score(h_refined)
            if new_val < best_local:
                best_local = new_val
                improved = True
            else:
                h_refined[idx] -= delta
        if not improved:
            break
    
    return h_refined, best_local

def _construct_from_spectral_decomposition():
    """
    Direct spectral approach: The circulant matrix C for autoconvolution has
    eigenvalues |H[k]|^2 / N. The all-ones vector is an eigenvector with
    eigenvalue |H[0]|^2 / N = (sum h)^2 / N.
    
    We want to minimize max eigenvalue / ((sum h)^2 / N) = max_k |H[k]|^2 / |H[0]|^2.
    
    This is equivalent to finding H that minimizes the condition number of the
    spectrum. The optimal solution for circular convolution is H[k] = 0 for k != 0,
    which gives h = delta (impulse), but that has sum = 1 and autoconv = delta,
    so ratio = 1/1 = 1.
    
    For linear convolution, it's different. Let's construct h directly to have
    minimal autoconvolution spread.
    """
    t_start = time.time()
    deadline = t_start + 30.0
    
    # Try various parametric families
    candidates = []
    
    # 1. Constant
    h_const = np.ones(N) / N
    candidates.append(h_const)
    
    # 2. Triangular (Bartlett) window
    h_tri = np.linspace(1, 0, N)
    h_tri = h_tri + np.linspace(0, 1, N)
    h_tri = h_tri / np.sum(h_tri)
    candidates.append(h_tri)
    
    # 3. Gaussian
    x = np.linspace(-3, 3, N)
    h_gauss = np.exp(-x**2 / 2)
    h_gauss = h_gauss / np.sum(h_gauss)
    candidates.append(h_gauss)
    
    # 4. Cosine window
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * np.arange(N) / N))
    h_cos = h_cos / np.sum(h_cos)
    candidates.append(h_cos)
    
    # 5. Alternating signs with decay
    for decay in [0.9, 0.95, 0.99]:
        h_alt = ((-1) ** np.arange(N)) * (decay ** np.arange(N))
        h_alt = h_alt / np.sum(h_alt) if abs(np.sum(h_alt)) > 1e-9 else np.ones(N)/N
        candidates.append(h_alt)
    
    # 6. Random with structure
    rng = np.random.default_rng(999)
    for _ in range(5):
        h_rand = rng.normal(0, 1, N)
        # Apply low-pass filter
        H = np.fft.rfft(h_rand)
        H[1:N//2] *= 0.1
        h_filtered = np.fft.irfft(H, N)
        h_filtered = h_filtered / np.sum(h_filtered) if abs(np.sum(h_filtered)) > 1e-9 else np.ones(N)/N
        candidates.append(h_filtered)
    
    best_h = candidates[0]
    best_score = _score(best_h)
    
    for h in candidates[1:]:
        if time.time() > deadline - 5:
            break
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Strategy 1: Spectral decomposition construction
    if time.time() < deadline - 50:
        h_spec, score_spec = _construct_from_spectral_decomposition()
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Strategy 2: Frequency-domain optimization
    if time.time() < deadline - 25:
        h_freq, score_freq = _spectral_optimize()
        if score_freq < best_score:
            best_score = score_freq
            best_h = h_freq.copy()
    
    # Strategy 3: Time-domain local refinement
    if time.time() < deadline - 5:
        h_ref, score_ref = _time_domain_refine(best_h)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
    
    return best_h

# EVOLVE-BLOCK-END
