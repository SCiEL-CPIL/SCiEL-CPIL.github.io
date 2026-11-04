"""
Spectral phase-retrieval strategy for signed autocorrelation maximization.

Strategy:
1. Model the problem in the frequency domain: fix |X(k)| to a constant (flat spectrum)
   and optimize only the phase vector phi.
2. The autoconvolution c = IFFT(|X|^2). With flat |X|, c = IFFT(const) = delta (ideal).
   We want to minimize the off-diagonal peaks of the *actual* autoconvolution,
   which corresponds to making the phase such that the time-domain signal has
   low sidelobes in its autoconvolution.
3. Use a smooth soft-max surrogate on the off-diagonal autoconvolution peaks,
   and optimize the phase vector via gradient descent in the spectral domain.
4. Convert back to time domain, ensuring the signal is real and signed.

Key insight: By working in the phase domain with fixed magnitude, we avoid the
non-convexity of jointly optimizing amplitude and phase in time domain.
"""
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

def _soft_max_abs(vals, beta=50.0):
    """Soft max of absolute values using log-sum-exp."""
    abs_vals = np.abs(vals)
    m = np.max(abs_vals)
    if m < 1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_vals - m)))) / beta

def _spectral_phase_objective(phi, beta=50.0):
    """
    Objective: minimize soft-max of off-diagonal autoconvolution peaks,
    given a fixed flat magnitude spectrum and variable phase phi.
    
    phi: complex phase vector (length N//2 + 1 for rfft)
    """
    # Construct the spectrum with fixed magnitude and variable phase
    # For a real signal, X(k) = A * exp(i*phi_k) where A is constant
    # We need to ensure conjugate symmetry for real output
    
    n = N
    half_n = n // 2 + 1
    
    # Fixed magnitude: choose A such that the DC component is reasonable
    # The integral S = sum(x) = X(0). We want S to be nonzero.
    # Let A = 1.0 for simplicity, then S = A * exp(i*phi_0)
    # To keep S real and positive, set phi_0 = 0
    
    mag = np.ones(half_n)
    # Ensure DC is real and positive
    mag[0] = 1.0
    
    # Phase: phi[0] = 0 (DC), phi[half_n-1] = 0 if n even (Nyquist)
    # For other bins, phase is free but must satisfy conjugate symmetry
    # In rfft representation, we just need to form the full spectrum
    
    # Build full complex spectrum
    full_mag = np.zeros(n, dtype=complex)
    full_mag[:half_n] = mag * np.exp(1j * phi[:half_n])
    
    # Enforce conjugate symmetry for real signal
    # For rfft, the full spectrum is implicitly defined
    # Let's use irfft directly
    
    # The phase vector phi should be length half_n
    # phi[0] and phi[-1] (if n even) must be 0 for real signal
    phase = np.zeros(half_n)
    phase[0] = 0.0
    if n % 2 == 0:
        phase[-1] = 0.0
    
    # Use the provided phi for the free bins
    free_start = 1
    free_end = half_n - 1 if n % 2 == 0 else half_n
    phase[free_start:free_end] = phi
    
    # Construct spectrum
    X = mag * np.exp(1j * phase)
    
    # Convert to time domain
    x = np.fft.irfft(X, n)
    
    # Compute autoconvolution
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    
    # Soft max of off-diagonal peaks
    return _soft_max_abs(c[1:], beta) / (s * s)

def _spectral_phase_gradient(phi, beta=50.0):
    """
    Numerical gradient of the spectral phase objective.
    Using central differences for stability.
    """
    eps = 1e-5
    grad = np.zeros_like(phi)
    for i in range(len(phi)):
        phi_plus = phi.copy()
        phi_minus = phi.copy()
        phi_plus[i] += eps
        phi_minus[i] -= eps
        f_plus = _spectral_phase_objective(phi_plus, beta)
        f_minus = _spectral_phase_objective(phi_minus, beta)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def _optimize_phase_spectral(phi_init, beta=50.0, max_iters=500, time_budget=60.0):
    """
    Optimize the phase vector using gradient descent with adaptive step size.
    """
    start = time.time()
    phi = phi_init.copy()
    best_phi = phi.copy()
    best_obj = _spectral_phase_objective(phi, beta)
    
    step_size = 0.1
    patience = 0
    max_patience = 50
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Compute gradient
        grad = _spectral_phase_gradient(phi, beta)
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        # Gradient descent step
        phi_new = phi - step_size * grad / gnorm
        
        # Evaluate new objective
        obj_new = _spectral_phase_objective(phi_new, beta)
        
        if obj_new < best_obj:
            best_obj = obj_new
            best_phi = phi_new.copy()
            phi = phi_new
            patience = 0
            # Increase step size slightly if improving
            step_size = min(step_size * 1.05, 0.5)
        else:
            patience += 1
            step_size *= 0.9
            if patience > max_patience:
                break
    
    return best_phi, best_obj

def _phase_to_signal(phi):
    """Convert phase vector to real time-domain signal."""
    half_n = N // 2 + 1
    mag = np.ones(half_n)
    
    phase = np.zeros(half_n)
    phase[0] = 0.0
    if N % 2 == 0:
        phase[-1] = 0.0
    
    free_start = 1
    free_end = half_n - 1 if N % 2 == 0 else half_n
    phase[free_start:free_end] = phi
    
    X = mag * np.exp(1j * phase)
    x = np.fft.irfft(X, N)
    
    # Ensure nonzero sum
    s = np.sum(x)
    if abs(s) < 1e-10:
        x += 1.0 / N
    
    return x

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    Uses spectral phase-retrieval: fix flat magnitude, optimize phase.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Generate multiple initial phase vectors
    half_n = N // 2 + 1
    free_start = 1
    free_end = half_n - 1 if N % 2 == 0 else half_n
    n_free = free_end - free_start
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # Candidate 1: Random phase
    phi_rand = rng.uniform(-np.pi, np.pi, n_free)
    candidates.append(phi_rand)
    
    # Candidate 2: Linearly varying phase (chirp)
    t = np.linspace(0, 2 * np.pi, n_free)
    phi_chirp = t
    candidates.append(phi_chirp)
    
    # Candidate 3: Alternating phase
    phi_alt = np.zeros(n_free)
    phi_alt[::2] = np.pi
    candidates.append(phi_alt)
    
    # Candidate 4: Sine phase
    phi_sine = np.sin(np.linspace(0, 2 * np.pi, n_free))
    candidates.append(phi_sine)
    
    # Candidate 5: Random with smaller scale
    phi_small = rng.uniform(-0.5, 0.5, n_free)
    candidates.append(phi_small)
    
    # Candidate 6: Golay-like phase pattern
    phi_golay = np.zeros(n_free)
    for i in range(n_free):
        # Use a simple algebraic pattern
        phi_golay[i] = np.pi * (i % 2)
    candidates.append(phi_golay)
    
    # Evaluate all candidates
    best_phi = None
    best_score = float('inf')
    best_x = None
    
    for phi in candidates:
        x = _phase_to_signal(phi)
        sc = _metric_offdiag(x)
        if sc < best_score:
            best_score = sc
            best_phi = phi.copy()
            best_x = x.copy()
    
    if best_x is None:
        return np.ones(N)
    
    # Phase 2: Spectral phase optimization with annealed beta
    phi_init = best_phi.copy()
    
    # Phase 2a: Coarse optimization with low beta
    try:
        if time.time() < deadline - 50.0:
            phi_coarse, obj_coarse = _optimize_phase_spectral(
                phi_init, beta=10.0, max_iters=300, time_budget=40.0
            )
            x_coarse = _phase_to_signal(phi_coarse)
            sc_coarse = _metric_offdiag(x_coarse)
            if sc_coarse < best_score:
                best_score = sc_coarse
                best_x = x_coarse.copy()
                phi_init = phi_coarse.copy()
    except Exception:
        pass
    
    # Phase 2b: Fine optimization with high beta
    try:
        if time.time() < deadline - 30.0:
            phi_fine, obj_fine = _optimize_phase_spectral(
                phi_init, beta=100.0, max_iters=400, time_budget=30.0
            )
            x_fine = _phase_to_signal(phi_fine)
            sc_fine = _metric_offdiag(x_fine)
            if sc_fine < best_score:
                best_score = sc_fine
                best_x = x_fine.copy()
    except Exception:
        pass
    
    # Phase 3: Local refinement in time domain with random perturbations
    x = best_x.copy()
    current_score = best_score
    rng2 = np.random.default_rng(777)
    
    max_iters = 200
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Random perturbation with decreasing scale
        scale = 0.05 * (1.0 - it / max_iters)
        mask = rng2.random(N) < 0.1
        perturb = rng2.normal(0, scale, N)
        x_new = x + mask * perturb
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-10:
            continue
        
        sc_new = _metric_offdiag(x_new)
        if sc_new < current_score:
            x = x_new
            current_score = sc_new
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
    
    # Phase 4: Try thresholding to ±1
    x_thresh = np.sign(best_x)
    if np.all(x_thresh == 1):
        x_thresh[::2] = -1
    elif np.all(x_thresh == -1):
        x_thresh[::2] = 1
    
    s_thresh = np.sum(x_thresh)
    if abs(s_thresh) > 1e-10:
        score_thresh = _metric_offdiag(x_thresh)
        if score_thresh < best_score:
            best_score = score_thresh
            best_x = x_thresh.copy()
    
    return best_x
