"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft, dct, idct
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
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

def _h_to_dct(h):
    """Convert time-domain signal to DCT-II coefficients."""
    return dct(h, type=2, norm='ortho')

def _dct_to_h(coeffs):
    """Convert DCT-II coefficients to time-domain signal."""
    return idct(coeffs, type=2, norm='ortho')

def _objective_spectral(c_ac, c_dc, beta, weight):
    """
    Combined objective: spectral flatness (L4 norm of DCT AC coeffs)
    plus soft-max peak of autoconvolution.
    """
    c_full = np.concatenate([[c_dc], c_ac])
    h_recon = _dct_to_h(c_full)
    
    if not np.all(np.isfinite(h_recon)):
        return 1e10
    
    s = np.sum(h_recon)
    if abs(s) < 1e-10:
        return 1e10
    
    h_norm = h_recon * (N / s)
    
    # L4 norm of AC DCT coefficients (spectral flatness proxy)
    c_norm = _h_to_dct(h_norm)
    l4_energy = np.sum(np.abs(c_norm[1:])**4)
    
    # Soft-max peak of autoconvolution
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    peak_term = abs(2.0 * N * sm / (N * N))
    
    return weight * l4_energy / (N**4) + (1.0 - weight) * peak_term

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Combines parametric families, random walks, spectral flattening in DCT domain,
    and multi-stage soft-max optimization with coordinate descent polish.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Quick parametric families to get a good starting point
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N)
    for m in range(1, 21):
        if time.time() > deadline - 80:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 2: Smoothed block (tanh)
    for beta_tanh in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 80:
            break
        h = np.tanh(beta_tanh * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Family 3: Two-level step function
    for frac in np.linspace(0.1, 0.9, 17):
        if time.time() > deadline - 80:
            break
        h = np.ones(N)
        n_high = int(frac * N)
        h[:n_high] = 1.5
        h[n_high:] = 0.5
        h = h - np.mean(h) + 1.0
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Family 4: Alternating patterns
    for period in [2, 3, 4, 5, 7, 8, 10]:
        if time.time() > deadline - 80:
            break
        for amp in [0.3, 0.5, 0.7]:
            h = 1.0 + amp * np.sin(2.0 * np.pi * np.arange(N) / period)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Phase 2: Random walk with zero-sum constraint
    if time.time() < deadline - 60:
        rng = np.random.RandomState(42)
        
        for noise_scale in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
            if time.time() > deadline - 60:
                break
            
            for trial in range(20):
                if time.time() > deadline - 60:
                    break
                
                noise = rng.randn(N) * noise_scale
                walk = np.cumsum(noise)
                
                # Subtract linear trend for zero-sum
                sum_i = np.sum(np.arange(N))
                drift = np.sum(walk) / sum_i
                h = 1.0 + walk - drift * np.arange(N)
                
                if not np.all(np.isfinite(h)) or abs(np.sum(h) - N) > 1e-6:
                    continue
                
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # Phase 3: Spectral flattening in DCT domain
    # This is the key insight: minimizing L4 norm of DCT coefficients flattens
    # the power spectrum, which reduces the peak of the autocorrelation
    c_best = _h_to_dct(best_h)
    c_dc = c_best[0].copy()

    # Multi-stage spectral flattening with varying weights
    for weight in [0.8, 0.6, 0.4, 0.2, 0.1]:
        if time.time() > deadline - 30:
            break
        
        def obj_spectral(c_ac, weight=weight, c_dc=c_dc):
            return _objective_spectral(c_ac, c_dc, beta=50.0, weight=weight)
        
        try:
            result = minimize(
                obj_spectral,
                c_best[1:].copy(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_ac_opt = result.x
            c_full_opt = np.concatenate([[c_dc], c_ac_opt])
            h_opt = _dct_to_h(c_full_opt)

            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _eval_value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
                    c_best = _h_to_dct(best_h)
                    c_dc = c_best[0].copy()
        except Exception:
            continue

    # Phase 4: Multi-stage soft-max optimization with increasing beta
    for beta in [2.0, 10.0, 50.0, 200.0, 500.0]:
        if time.time() > deadline - 15:
            break
        
        def obj_soft(c_ac, beta=beta, c_dc=c_dc):
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = _dct_to_h(c_full)
            if not np.all(np.isfinite(h_recon)):
                return 1e10
            s = np.sum(h_recon)
            if abs(s) < 1e-10:
                return 1e10
            h_norm = h_recon * (N / s)
            conv = _convolve_fft(h_norm)
            sm = _soft_max(conv, beta)
            return abs(2.0 * N * sm / (N * N))
        
        rng = np.random.RandomState(42 + int(beta))
        c_ac_init = c_best[1:].copy()

        for trial in range(3):
            if time.time() > deadline - 15:
                break

            if trial == 0:
                c_ac_trial = c_ac_init.copy()
            else:
                c_ac_trial = c_ac_init + 0.05 * rng.randn(N-1)

            try:
                result = minimize(
                    obj_soft,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
                        c_ac_init = c_best[1:].copy()
            except Exception:
                continue

    # Phase 5: Final high-precision polish with very high beta
    if time.time() < deadline - 5:
        beta_final = 1000.0
        
        def obj_final(c_ac, beta=beta_final, c_dc=c_dc):
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = _dct_to_h(c_full)
            if not np.all(np.isfinite(h_recon)):
                return 1e10
            s = np.sum(h_recon)
            if abs(s) < 1e-10:
                return 1e10
            h_norm = h_recon * (N / s)
            conv = _convolve_fft(h_norm)
            sm = _soft_max(conv, beta)
            return abs(2.0 * N * sm / (N * N))
        
        try:
            result = minimize(
                obj_final,
                c_best[1:].copy(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            c_ac_opt = result.x
            c_full_opt = np.concatenate([[c_dc], c_ac_opt])
            h_opt = _dct_to_h(c_full_opt)
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _eval_value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
