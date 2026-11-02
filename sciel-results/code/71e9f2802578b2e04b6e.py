"""Spectral-domain optimization: PAPR minimization via alternating phase/amplitude + L-BFGS polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _fft_len(n):
    L = 1
    while L < n:
        L *= 2
    return L

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = _fft_len(L)
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _compute_objective(x):
    """Compute the raw objective: max|ac| / (sum x)^2"""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_ac_and_grad(x, beta):
    """
    Compute soft-max of |autoconvolution| and its gradient w.r.t. x.
    Uses FFT-based gradient computation.
    """
    L = 2 * N - 1
    L_fft = _fft_len(L)
    
    X = np.fft.rfft(x, n=L_fft)
    ac = np.fft.irfft(X * X, n=L_fft)[:L]
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    if beta < 1e6:
        exps = np.exp(beta * (abs_ac - m))
        log_sum = np.log(np.sum(exps))
        soft_val = m + log_sum / beta
        weights = exps / np.sum(exps)
    else:
        soft_val = m
        weights = np.zeros(L)
        weights[np.argmax(abs_ac)] = 1.0
    
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length L = 2N-1
    
    # Gradient w.r.t. x: d/dx <grad_ac, ac> where ac = x * x (linear conv)
    # ac[k] = sum_i x[i] * x[k-i]
    # d<grad_ac, ac>/dx[j] = sum_k grad_ac[k] * (x[k-j] + x[j-k]) for valid k
    # This is: grad_x = grad_ac * x + flip(grad_ac) * x (cross-correlation style)
    # Using FFT: grad_x = IFFT(FFT(grad_ac) * FFT(x)) + IFFT(flip(FFT(grad_ac)) * FFT(x))
    # Actually simpler: grad_x[j] = sum_k grad_ac[k] * (x[k-j] + x[j-k])
    # = (grad_ac * x)[j] + (grad_ac_reversed * x)[j] where appropriate
    
    # Direct computation via FFT:
    # ac = IFFT(FFT(x)^2)
    # d ac / dx = 2 * IFFT(FFT(x) * FFT(dx))  -- but this is for circular
    # For linear: we need to be careful
    
    # Use the fact that: <g, x*x> = <g*x + flip(g)*x, x> (for linear conv with zero padding)
    # So grad_x = g * x + flip(g) * x (linear convolutions)
    
    L_corr = _fft_len(3 * N)
    G = np.fft.rfft(grad_ac, n=L_corr)
    Xf = np.fft.rfft(x, n=L_corr)
    
    # term1: grad_ac convolved with x
    term1 = np.fft.irfft(G * Xf, n=L_corr)[:N]
    # term2: flip(grad_ac) convolved with x
    grad_ac_flip = grad_ac[::-1]
    Gf = np.fft.rfft(grad_ac_flip, n=L_corr)
    term2 = np.fft.irfft(Gf * Xf, n=L_corr)[:N]
    
    grad_x = term1 + term2
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient for L-BFGS."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    soft_val, grad_soft = _soft_max_ac_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_papr_optimize(x_init, iters=30, beta=50.0):
    """
    Optimize in spectral domain: alternate between phase optimization (PAPR reduction)
    and amplitude optimization.
    
    Key insight: ac = IFFT(|X|^2). We want to minimize max|IFFT(|X|^2)|.
    This is the PAPR problem. We optimize the phase of X to minimize PAPR,
    then optimize |X| to further reduce the normalized objective.
    """
    L = 2 * N - 1
    L_fft = _fft_len(L)
    
    X = np.fft.rfft(x_init, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    mag[0] = np.abs(X[0])
    phase[0] = 0.0
    
    best_x = x_init.copy()
    best_val = _compute_objective(x_init)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        # Phase optimization: minimize max|IFFT(|X|^2)| by adjusting phase
        # Gradient of soft-max PAPR w.r.t. phase
        for _ in range(3):  # inner phase iterations
            if _check_deadline():
                break
            
            spectrum = mag * np.exp(1j * phase)
            spectrum[0] = mag[0]  # DC is real
            x = np.fft.irfft(spectrum, n=L_fft)[:N]
            
            s = np.sum(x)
            if abs(s) < 1e-10:
                break
            x_norm = x / s
            
            # Compute gradient of soft-max |ac| w.r.t. x
            val, grad_x = _soft_max_ac_and_grad(x_norm, beta)
            
            # Gradient w.r.t. phase: d<grad_x, x>/dphase_k
            # x = IFFT(mag * exp(i*phase))
            # dx/dphase_k = (1/L_fft) * i * mag[k] * exp(i*phase_k) * basis_k
            # <grad_x, dx/dphase_k> = Re[conj(i * mag[k] * exp(i*phase_k)) * conj(FFT(grad_x)_k)] / L_fft
            # = Re[-i * conj(mag[k] * exp(i*phase_k)) * conj(G_k)] / L_fft
            # = Re[-i * conj(X_k) * conj(G_k)] / L_fft
            # = Re[i * X_k * G_k] / L_fft  (for k > 0, accounting for real FFT symmetry)
            
            G = np.fft.rfft(grad_x, n=L_fft)
            
            # For real FFT: X[k] = mag[k] * exp(i*phase[k])
            # grad_phase[k] = Re[i * X[k] * G[k]] / L_fft for k=1..L_fft/2-1
            # grad_phase[0] = 0 (phase of DC is fixed at 0)
            # grad_phase[L_fft/2] = 0 (Nyquist, real)
            
            grad_phase = np.zeros_like(phase)
            if L_fft > 2:
                # i * X * G = i * mag * exp(i*phase) * G
                iXG = 1j * X * G
                grad_phase[1:L_fft//2] = np.real(iXG[1:L_fft//2]) / L_fft
            
            # Gradient descent on phase
            step = 1.0 / (beta * np.max(mag) + 1e-6)
            phase = phase - step * grad_phase
            phase[0] = 0.0
            if L_fft % 2 == 0:
                phase[L_fft // 2] = 0.0  # Nyquist
        
        # Reconstruct x and check objective
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_new = np.fft.irfft(spectrum, n=L_fft)[:N]
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-10:
            x_new = x_new / s_new
        
        val_new = _compute_objective(x_new)
        if val_new < best_val:
            best_val = val_new
            best_x = x_new.copy()
            # Update X from best_x
            X = np.fft.rfft(x_new, n=L_fft)
            mag = np.abs(X)
            phase = np.angle(X)
            mag[0] = np.abs(X[0])
            phase[0] = 0.0
        
        # Amplitude optimization: adjust mag to reduce objective
        if it % 3 == 2:  # Every 3rd iteration
            if _check_deadline():
                break
            
            # Gradient w.r.t. amplitude
            spectrum = mag * np.exp(1j * phase)
            spectrum[0] = mag[0]
            x = np.fft.irfft(spectrum, n=L_fft)[:N]
            s = np.sum(x)
            if abs(s) < 1e-10:
                continue
            x_norm = x / s
            
            val, grad_x = _soft_max_ac_and_grad(x_norm, beta)
            G = np.fft.rfft(grad_x, n=L_fft)
            
            # grad_mag[k] = Re[conj(X[k]) * G[k]] / L_fft for k>0
            # For real FFT with symmetry, factor of 2 for non-DC, non-Nyquist
            grad_mag = np.zeros_like(mag)
            if L_fft > 1:
                grad_mag[1:] = (2.0 / L_fft) * np.real(np.conj(X[1:]) * G[1:])
            grad_mag[0] = np.real(G[0]) / L_fft
            
            step = 1.0 / (beta * np.max(mag) + 1e-6)
            mag = mag - step * grad_mag
            mag = np.maximum(mag, 0.01)
            mag[0] = np.maximum(mag[0], 0.01)
    
    return best_x

def _spectral_init(rng, mode=0):
    """Generate diverse spectral initializations."""
    L = 2 * N - 1
    L_fft = _fft_len(L)
    
    if mode == 0:
        # Flat spectrum
        mag = np.ones(L_fft // 2 + 1)
        freqs = np.fft.rfftfreq(L_fft)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        # Raised cosine
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        # Low-pass
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-5.0 * (freqs * L_fft / 2) ** 2)
    elif mode == 3:
        # Bandpass
        freqs = np.fft.rfftfreq(L_fft)
        f = np.abs(freqs * L_fft / 2)
        mag = np.exp(-2.0 * (f - 0.3) ** 2)
    else:
        # Random smooth spectrum
        mag = 1.0 + 0.5 * rng.normal(0, 1, size=L_fft // 2 + 1)
        # Smooth it
        from scipy.ndimage import uniform_filter1d
        mag = uniform_filter1d(mag, size=10)
        mag = np.maximum(0.1, mag)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0.0
    if L_fft % 2 == 0:
        phase[L_fft // 2] = 0.0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x / s
    return x

def _lbfgs_polish(x_init, beta, maxiter=200):
    """L-BFGS polish with given beta."""
    try:
        def obj_and_grad(x, _beta=beta):
            return _objective_and_grad(x, _beta)
        
        result = minimize(obj_and_grad, x_init, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12})
        return result.x, result.fun
    except Exception:
        return x_init, _compute_objective(x_init)

def _coordinate_polish(x, max_iter=5000):
    """Coordinate descent polish with adaptive step size."""
    best_x = x.copy()
    best_val = _compute_objective(x)
    step_size = 0.02
    no_improve = 0
    rng = np.random.default_rng(123)
    
    for _ in range(max_iter):
        if _check_deadline():
            break
        if no_improve > 2000:
            break
        
        # Perturb a small random subset
        n_perturb = rng.integers(1, 6)
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        
        val = _compute_objective(x_trial)
        if val < best_val - 1e-12:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.002, step_size * 0.97)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve % 100 == 0:
                step_size = min(0.05, step_size * 1.03)
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _compute_objective(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = [np.ones(N)]
    
    # Spectral initializations with various spectra
    for mode in range(5):
        for _ in range(3):
            if _check_deadline():
                break
            initial_points.append(_spectral_init(rng, mode))
    
    # Some time-domain structured initializations
    t = np.linspace(0, 1, N)
    initial_points.extend([
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        np.exp(1.0 * t),
        np.exp(-1.0 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
    ])
    
    # Random initializations
    for _ in range(5):
        if _check_deadline():
            break
        initial_points.append(rng.normal(1.0, 0.2, size=N) * rng.uniform(0.5, 1.5))
    
    candidates = []
    
    # Phase 1: Spectral PAPR optimization from each initial point
    for x0 in initial_points:
        if _check_deadline():
            break
        
        try:
            x_spec = _spectral_papr_optimize(x0, iters=15, beta=30.0)
            val = _compute_objective(x_spec)
            candidates.append((val, x_spec))
            if val < best_val:
                best_val = val
                best_x = x_spec.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS polish on top candidates
    candidates.sort(key=lambda item: item[0])
    top_candidates = [x for _, x in candidates[:8]]
    
    for x_init in top_candidates:
        if _check_deadline():
            break
        
        for beta in [50.0, 150.0]:
            if _check_deadline():
                break
            try:
                x_pol, val_pol = _lbfgs_polish(x_init, beta, maxiter=150)
                candidates.append((val_pol, x_pol))
                if val_pol < best_val:
                    best_val = val_pol
                    best_x = x_pol.copy()
            except Exception:
                pass
    
    # Phase 3: Re-sort and do high-precision polish
    candidates.sort(key=lambda item: item[0])
    top2 = [x for _, x in candidates[:5]]
    
    for x_init in top2:
        if _check_deadline():
            break
        
        for beta in [300.0, 800.0]:
            if _check_deadline():
                break
            try:
                x_pol, val_pol = _lbfgs_polish(x_init, beta, maxiter=250)
                if val_pol < best_val:
                    best_val = val_pol
                    best_x = x_pol.copy()
            except Exception:
                pass
    
    # Phase 4: Coordinate descent polish
    if not _check_deadline():
        x_cd, val_cd = _coordinate_polish(best_x, max_iter=4000)
        if val_cd < best_val:
            best_val = val_cd
            best_x = x_cd.copy()
    
    # Phase 5: Final spectral polish
    if not _check_deadline():
        try:
            x_final = _spectral_papr_optimize(best_x, iters=20, beta=100.0)
            val_final = _compute_objective(x_final)
            if val_final < best_val:
                best_val = val_final
                best_x = x_final.copy()
        except Exception:
            pass
    
    # Phase 6: One more coordinate polish
    if not _check_deadline():
        x_cd2, val_cd2 = _coordinate_polish(best_x, max_iter=3000)
        if val_cd2 < best_val:
            best_val = val_cd2
            best_x = x_cd2.copy()
    
    return best_x
# EVOLVE-BLOCK-END
