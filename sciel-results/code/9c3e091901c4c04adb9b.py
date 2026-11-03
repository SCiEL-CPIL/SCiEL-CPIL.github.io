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

def _freq_objective(x):
    """
    Objective in frequency domain using spectral decomposition.
    x[0] = H[0].real (DC, must be real for real h)
    x[1] = H[1].real, x[2] = H[1].imag
    ...
    x[2k-1] = H[k].real, x[2k] = H[k].imag for k=1..N/2
    
    We minimize max|autoconv_linear| / (sum h)^2 = max|autoconv_linear| / H[0]^2
    where autoconv_linear is computed via zero-padding.
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
    Optimize using spectral decomposition of the circulant structure.
    Key insight: the autoconvolution ratio can be expressed in terms of the
    frequency components. We directly optimize the frequency coefficients.
    """
    t_start = time.time()
    deadline = t_start + 85.0
    
    n_half = N // 2
    n_vars = 1 + 2 * n_half
    
    # Initial guess: H[0]=1, others small
    x0 = np.zeros(n_vars)
    x0[0] = 1.0
    for k in range(1, n_half + 1):
        x0[2*k - 1] = 1.0 / N
        x0[2*k] = 0.0
    
    best_x = x0.copy()
    best_score = _freq_objective(x0)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Flat
    initializations.append(x0.copy())
    
    # 2. Various decay profiles
    for alpha in [0.3, 0.7, 1.0, 1.5, 2.0, 3.0]:
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        for k in range(1, n_half + 1):
            x_init[2*k - 1] = 1.0 / (1.0 + k**alpha)
            x_init[2*k] = 0.0
        initializations.append(x_init)
    
    # 3. Cosine taper
    for beta in [0.5, 1.0, 2.0]:
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        for k in range(1, n_half + 1):
            x_init[2*k - 1] = np.cos(np.pi * k / (2 * n_half)) ** beta
            x_init[2*k] = 0.0
        initializations.append(x_init)
    
    # 4. Random perturbations
    rng = np.random.default_rng(12345)
    for _ in range(5):
        x_init = x0.copy()
        x_init[1:] += rng.normal(0, 0.05, size=2*n_half)
        initializations.append(x_init)
    
    # 5. Sparse spectrum: only few non-zero frequencies
    for n_nonzero in [3, 5, 8, 12]:
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        indices = rng.choice(range(1, n_half + 1), size=min(n_nonzero, n_half), replace=False)
        for idx in indices:
            x_init[2*idx - 1] = rng.uniform(0.1, 0.5)
            x_init[2*idx] = rng.uniform(-0.1, 0.1)
        initializations.append(x_init)
    
    for x_init in initializations:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _freq_objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Convert to time domain
    H = np.zeros(N, dtype=complex)
    H[0] = best_x[0]
    for k in range(1, n_half + 1):
        H[k] = best_x[2*k - 1] + 1j * best_x[2*k]
        H[N - k] = np.conj(H[k])
    
    h = np.fft.ifft(H).real
    
    # Local refinement in frequency domain
    if time.time() < deadline - 10:
        x_local = best_x.copy()
        best_local = best_score
        
        for iteration in range(20):
            improved = False
            for idx in range(1, n_vars):
                if time.time() > deadline - 3:
                    break
                for delta in [0.005, -0.005, 0.02, -0.02, 0.05, -0.05]:
                    x_local[idx] += delta
                    new_val = _freq_objective(x_local)
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
            H = np.zeros(N, dtype=complex)
            H[0] = best_x[0]
            for k in range(1, n_half + 1):
                H[k] = best_x[2*k - 1] + 1j * best_x[2*k]
                H[N - k] = np.conj(H[k])
            h = np.fft.ifft(H).real
    
    return h, best_score

def _time_refine(h):
    """Local search in time domain."""
    t_start = time.time()
    deadline = t_start + 20.0
    
    h_refined = h.copy()
    best_local = _score(h_refined)
    
    # Sign flips
    for _ in range(8):
        improved = False
        for idx in range(N):
            if time.time() > deadline - 3:
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
    
    # Small perturbations
    rng = np.random.default_rng(777)
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        idx = rng.integers(0, N)
        delta = rng.normal(0, 0.02)
        h_refined[idx] += delta
        new_val = _score(h_refined)
        if new_val < best_local:
            best_local = new_val
        else:
            h_refined[idx] -= delta
    
    return h_refined, best_local

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Spectral optimization
    if time.time() < deadline - 25:
        h_spec, score_spec = _spectral_optimize()
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Time-domain refinement
    if time.time() < deadline - 3:
        h_ref, score_ref = _time_refine(best_h)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
    
    return best_h

# EVOLVE-BLOCK-END
