"""Combined program: discrete chirp (z^k) representation + spectral flattening + soft-max polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft, dct, idct
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value exactly."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective_exact(h):
    """Exact objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_soft(h, beta):
    """Soft-max surrogate objective."""
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _chirp_h(theta, a):
    """
    Build h from a discrete chirp: h[k] = 1 + a * cos(theta * k).
    This is the real part of (1 + a * z^k) where z = e^{i*theta}.
    The autoconvolution peak is bounded by the Dirichlet kernel,
    which for non-integer multiples of 2*pi/N decays as 1/|sin(theta/2)|.
    """
    k = np.arange(N)
    return 1.0 + a * np.cos(theta * k)

def _chirp_objective(params):
    """
    Objective for chirp parameterization: params = [theta, a].
    """
    theta, a = params
    h = _chirp_h(theta, a)
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _dct_coeffs(h):
    """Convert to DCT-II coefficients."""
    return dct(h, type=2, norm='ortho')

def _from_dct_coeffs(c):
    """Convert from DCT-II coefficients."""
    return idct(c, type=2, norm='ortho')

def _spectral_flatness_l4(c):
    """L4 norm of AC DCT coefficients (proxy for spectral flatness)."""
    return np.sum(np.abs(c[1:] ** 4))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Combines discrete chirp representation with spectral flattening and soft-max polish.
    """
    deadline = time.time() + 145.0
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Discrete chirp search over (theta, a)
    # The key insight: h[k] = 1 + a*cos(theta*k) has autoconvolution peak
    # bounded by Dirichlet kernel magnitude, which is much smaller than N
    # for generic theta (non-integer multiples of 2*pi/N).
    if time.time() < deadline - 90:
        # Grid search over theta and a
        # theta should avoid multiples of 2*pi/N to avoid aliasing with DC
        # Good theta values are irrational multiples of pi or random
        theta_vals = np.linspace(0.01, np.pi - 0.01, 80)
        a_vals = np.linspace(-1.5, 1.5, 31)
        
        for theta in theta_vals:
            if time.time() > deadline - 90:
                break
            for a in a_vals:
                h = _chirp_h(theta, a)
                v = _objective_exact(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
        
        # Also try random theta values (irrational-like)
        rng = np.random.RandomState(42)
        for _ in range(50):
            if time.time() > deadline - 85:
                break
            theta = rng.uniform(0.001, np.pi - 0.001)
            a = rng.uniform(-1.5, 1.5)
            h = _chirp_h(theta, a)
            v = _objective_exact(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Phase 2: Local optimization of chirp parameters
    if time.time() < deadline - 70:
        # Extract initial theta, a from best_h via fitting
        k = np.arange(N)
        # Fit h ~ 1 + a*cos(theta*k) using least squares
        # Try multiple initial theta values
        for theta0 in np.linspace(0.05, np.pi - 0.05, 20):
            if time.time() > deadline - 65:
                break
            try:
                result = minimize(
                    _chirp_objective,
                    [theta0, 0.5],
                    method='L-BFGS-B',
                    bounds=[(0.001, np.pi - 0.001), (-3.0, 3.0)],
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_opt = _chirp_h(result.x[0], result.x[1])
                if np.all(np.isfinite(h_opt)):
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue

    # Phase 3: DCT-domain spectral flattening
    # Convert best_h to DCT coefficients and minimize L4 norm of AC components
    if time.time() < deadline - 40:
        c_best = _dct_coeffs(best_h)
        c_dc = c_best[0].copy()
        
        for weight in [0.9, 0.7, 0.5]:
            if time.time() > deadline - 35:
                break
            
            def obj_flat(c_ac, c_dc=c_dc, weight=weight):
                c_full = np.concatenate([[c_dc], c_ac])
                h_recon = _from_dct_coeffs(c_full)
                if not np.all(np.isfinite(h_recon)):
                    return 1e10
                s = np.sum(h_recon)
                if abs(s) < 1e-10:
                    return 1e10
                h_norm = h_recon * (N / s)
                c_norm = _dct_coeffs(h_norm)
                l4_energy = np.sum(np.abs(c_norm[1:] ** 4))
                conv = _convolve_fft(h_norm)
                sm = _soft_max(conv, 50.0)
                peak_term = abs(2.0 * N * sm / (N * N))
                return weight * l4_energy / (N ** 4) + (1.0 - weight) * peak_term
            
            try:
                result = minimize(
                    obj_flat,
                    c_best[1:].copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                c_full_opt = np.concatenate([[c_dc], result.x])
                h_opt = _from_dct_coeffs(c_full_opt)
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue

    # Phase 4: Soft-max polish in time domain with multiple betas
    if time.time() < deadline - 15:
        for beta in [50.0, 200.0, 500.0]:
            if time.time() > deadline - 10:
                break
            
            def obj_polish(h, beta=beta):
                return _objective_soft(h, beta)
            
            rng = np.random.RandomState(100 + int(beta))
            for trial in range(4):
                if time.time() > deadline - 10:
                    break
                
                if trial == 0:
                    h_init = best_h.copy()
                else:
                    h_init = best_h + 0.02 * rng.randn(N)
                    h_init -= np.mean(h_init)
                    h_init += 1.0
                    h_init = h_init * (N / np.sum(h_init))
                
                try:
                    result = minimize(
                        obj_polish,
                        h_init,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
                    )
                    h_opt = result.x
                    if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                        v_opt = _objective_exact(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                except Exception:
                    continue

    # Phase 5: Coordinate descent polish
    if time.time() < deadline - 8:
        h = best_h.copy()
        step = 0.05
        for iteration in range(60):
            if time.time() > deadline - 5:
                break
            improved = False
            idx = np.argsort(np.abs(h))[::-1]
            for i in idx:
                if time.time() > deadline - 5:
                    break
                old = h[i]
                for delta in [step, -step]:
                    h[i] = old + delta
                    s = np.sum(h)
                    if abs(s) < 1e-10:
                        h[i] = old
                        continue
                    v = _objective_exact(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
