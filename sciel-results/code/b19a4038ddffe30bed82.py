import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    """Metric: max |c_k| for k != 0 / S^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _soft_max_abs(c, beta):
    """Smooth approximation to max|c| using log-sum-exp."""
    abs_c = np.abs(c)
    if len(abs_c) == 0:
        return 0.0
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta

def _objective_soft(x, beta, start_time, deadline):
    """Objective function for L-BFGS-B with time guard."""
    if time.time() > deadline:
        return 1e10
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    soft_max = _soft_max_abs(abs_c, beta)
    return soft_max / (s * s)

def _frft(x, alpha):
    """
    Fractional Fourier Transform of x.
    alpha is the fractional order (in units of pi/2).
    For alpha=0: identity
    For alpha=1: FFT (scaled)
    For alpha=2: time reversal
    We use the standard definition with proper scaling.
    """
    n = len(x)
    # FRFT kernel: K_alpha(t, u)
    # We use the eigenfunction decomposition approach or direct formula
    # For simplicity, use the property: FRFT^alpha = exp(i*alpha*pi/2 * (tD + Dt))
    # where D is the Fourier transform operator.
    # A practical implementation:
    # x_alpha(t) = sum_u K_alpha(t,u) x(u)
    # K_alpha(t,u) = (1/sqrt(2*pi)) * exp(-i*alpha*pi/4) * exp(-i*|cos(alpha*pi/2)|*t*u) * exp(-i*(1-|cos(alpha*pi/2)|)*(t^2+u^2)/2)
    
    # Actually, let's use a simpler and more robust approach:
    # FRFT can be computed as:
    # If alpha is close to integer, use direct formula
    # Otherwise, use the matrix representation
    
    # For our purposes, we'll use a parametric approach:
    # x_FRFT = cos(alpha*pi/2) * x - i * sin(alpha*pi/2) * FFT(x)
    # This is valid for alpha in [0, 1] and can be extended.
    
    # More accurately, the FRFT of order alpha (where alpha is in radians) is:
    # F_alpha(x)(t) = sum_u K_alpha(t,u) x(u)
    # where K_alpha is the FRFT kernel.
    
    # Let's use a numerically stable approach via the eigendecomposition of the DFT matrix,
    # but that's expensive. Instead, use the approximation:
    # For small alpha, FRFT ≈ x + i*alpha*pi/2 * (t*x(t) - i*FFT stuff)
    
    # Actually, the cleanest way for our optimization is to parameterize the sequence
    # directly in the time-frequency domain and use the FRFT as a mixing operation.
    
    # Let's implement the standard FRFT formula:
    a = alpha * np.pi / 2.0  # Convert to radians
    ca = np.cos(a)
    sa = np.sin(a)
    
    if abs(ca) < 1e-10:
        # alpha ≈ 1 (or 3): pure Fourier transform
        return np.fft.fft(x)
    
    if abs(sa) < 1e-10:
        # alpha ≈ 0: identity
        return x.copy()
    
    if abs(ca + 1) < 1e-10:
        # alpha ≈ 2: time reversal
        return x[::-1].copy()
    
    # General case: use the integral formula discretized
    n = len(x)
    t = np.arange(n, dtype=float)
    
    # FRFT kernel
    # K(t,u) = (1/sqrt(2*pi)) * exp(-i*a/4) * exp(-i*|ca|*t*u) * exp(-i*(1-|ca|)*(t^2+u^2)/2)
    # This is for continuous FRFT. For discrete, we need to be careful.
    
    # Let's use a simpler discrete FRFT definition based on the DFT eigenvectors.
    # The DFT matrix F has eigenvectors with eigenvalues 1, -i, -1, i.
    # FRFT = F @ diag(exp(i*alpha*pi/2 * eigenvalues)) @ F^{-1}
    # But this is O(n^2) or O(n log n) with FFT tricks.
    
    # For efficiency, let's use the approximation:
    # x_FRFT ≈ ca * x - 1j * sa * np.fft.fft(x) / sqrt(n)
    # This is the "linear" approximation which is exact for alpha in {0, 1}
    # and a good approximation for intermediate values.
    
    fft_x = np.fft.fft(x)
    return ca * x - 1j * sa * fft_x / np.sqrt(n)

def _generate_frft_candidates(deadline, max_candidates=12):
    """
    Generate candidates using FRFT mixing of random phase signals.
    """
    candidates = []
    rng = np.random.default_rng(12345)
    
    for i in range(max_candidates):
        if time.time() > deadline - 20.0:
            break
        
        # Random phase flat spectrum signal
        size = 1
        while size < 2 * N - 1:
            size *= 2
        phases = rng.uniform(-np.pi, np.pi, size // 2 + 1)
        mag = np.ones(size // 2 + 1)
        spectrum = mag * np.exp(1j * phases)
        spectrum[0] = mag[0]
        if size % 2 == 0:
            spectrum[-1] = mag[-1]
        x_full = np.fft.irfft(spectrum, size)
        x_base = x_full[:N]
        
        # Apply FRFT with random alpha
        alpha = rng.uniform(0.1, 0.9)
        x_frft = _frft(x_base, alpha)
        
        # Take real part and add small imaginary to make it complex
        # But we need real-valued output. So let's work with real signals only.
        # Actually, the task says "signed step heights" which implies real values.
        # So we should ensure our signals are real.
        
        # Let's use a different approach: generate real signals with random phases
        # in the frequency domain, then apply a time-frequency mixing.
        
        # Alternative: use the real part of the FRFT
        x_real = np.real(x_frft)
        
        # Normalize
        s = np.sum(x_real)
        if abs(s) > 1e-10:
            x_real = x_real / s * np.sqrt(N)
        
        candidates.append(x_real)
    
    return candidates

def _generate_chirp_candidates(deadline, max_candidates=8):
    """
    Generate chirp-based candidates.
    """
    candidates = []
    rng = np.random.default_rng(99999)
    
    for i in range(max_candidates):
        if time.time() > deadline - 15.0:
            break
        
        t = np.arange(N, dtype=float)
        
        # Linear chirp
        f0 = rng.uniform(0.5, 2.0)
        f1 = rng.uniform(2.0, N/4)
        f = f0 + (f1 - f0) * t / N
        
        # Add random phase
        phase = rng.uniform(0, 2*np.pi)
        
        x = np.cos(2 * np.pi * np.cumsum(f) / N + phase)
        
        # Add small random perturbation
        x += 0.1 * rng.normal(0, 1, N)
        
        candidates.append(x)
    
    return candidates

def _generate_binary_candidates(deadline, max_candidates=4):
    """
    Generate binary +/-1 candidates.
    """
    candidates = []
    rng = np.random.default_rng(77777)
    
    for i in range(max_candidates):
        if time.time() > deadline - 10.0:
            break
        
        x = rng.choice([-1, 1], N)
        candidates.append(x.astype(float))
    
    return candidates

def _refine_lbfgs(x_init, deadline, beta_start=10.0, beta_end=80.0, n_anneal_steps=4):
    """
    Refine x_init using L-BFGS-B with annealed soft-max beta.
    """
    x = x_init.copy()
    
    for step in range(n_anneal_steps):
        if time.time() > deadline - 3.0:
            break
        
        t_anneal = step / max(1, n_anneal_steps - 1)
        beta = beta_start + (beta_end - beta_start) * t_anneal
        
        def obj(x):
            return _objective_soft(x, beta, time.time(), deadline)
        
        try:
            result = scipy.optimize.minimize(
                obj,
                x,
                method='L-BFGS-B',
                options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-8},
                tol=1e-10
            )
            x = result.x
        except Exception:
            pass
    
    return x

def _polish_subgradient(x, deadline, max_iters=30):
    """
    Polish using subgradient descent on the exact max |c_k|.
    """
    x = x.copy()
    best_x = x.copy()
    best_score = _metric_offdiag(x)
    
    for i in range(max_iters):
        if time.time() > deadline - 1.0:
            break
            
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
            
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        
        # Find active lags (top 15%)
        threshold = np.percentile(abs_c, 85)
        active_mask = abs_c > threshold
        if not np.any(active_mask):
            break
            
        grad = np.zeros_like(x)
        active_indices = np.where(active_mask)[0]
        
        for k_idx in active_indices:
            k = k_idx + 1
            sign_c = np.sign(c[k])
            if sign_c == 0:
                sign_c = 1.0
            
            if k < N:
                grad[:N-k] += sign_c * x[k:]
                grad[k:] += sign_c * x[:N-k]
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
            
        step_size = 0.005 * (1.0 - i/max_iters) + 0.001
        x_new = x - step_size * grad / gnorm
        
        sc_new = _metric_offdiag(x_new)
        
        if sc_new < best_score:
            best_score = sc_new
            best_x = x_new.copy()
            x = x_new
        else:
            step_size *= 0.5
            x_new = x - step_size * grad / gnorm
            sc_new = _metric_offdiag(x_new)
            if sc_new < best_score:
                best_score = sc_new
                best_x = x_new.copy()
                x = x_new
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_x = None
    best_score = float('inf')
    
    def update_best(x, score):
        nonlocal best_x, best_score
        if x is not None and score < best_score:
            best_score = score
            best_x = x.copy()
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # FRFT-based candidates
    candidates.extend(_generate_frft_candidates(deadline, max_candidates=10))
    
    # Chirp-based candidates
    candidates.extend(_generate_chirp_candidates(deadline, max_candidates=6))
    
    # Binary candidates
    candidates.extend(_generate_binary_candidates(deadline, max_candidates=3))
    
    # Random noise
    rng = np.random.default_rng(42)
    candidates.append(rng.normal(0, 1, N))
    
    # Phase 2: Refine each candidate
    for i, x_init in enumerate(candidates):
        if time.time() > deadline - 15.0:
            break
        
        sc_init = _metric_offdiag(x_init)
        update_best(x_init, sc_init)
        
        x_ref = _refine_lbfgs(x_init, deadline)
        sc_ref = _metric_offdiag(x_ref)
        update_best(x_ref, sc_ref)
        
        if time.time() < deadline - 5.0:
            x_pol, sc_pol = _polish_subgradient(x_ref, deadline)
            update_best(x_pol, sc_pol)
    
    # Final polish on the best
    if best_x is not None and time.time() < deadline - 2.0:
        x_final, sc_final = _polish_subgradient(best_x, deadline, max_iters=15)
        update_best(x_final, sc_final)
    
    if best_x is None:
        return np.ones(N)
        
    return best_x
