"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import dct, idct, fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT (faster than np.convolve for large N)."""
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

def _objective_soft(h, beta):
    """
    Soft-max surrogate objective for autoconvolution peak.
    h is normalized internally so sum(h) = N.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _objective_exact(h):
    """Exact objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _dct_to_h(coeffs):
    """Convert DCT-II coefficients to time-domain signal."""
    return idct(coeffs, type=2, norm='ortho')

def _h_to_dct(h):
    """Convert time-domain signal to DCT-II coefficients."""
    return dct(h, type=2, norm='ortho')

def _spike_objective(params, n_spikes, spike_indices):
    """
    Objective for spike parameterization.
    params = [bg, a_1, a_2, ..., a_{n_spikes}]
    h = bg + sum_j a_j * delta_{j} (where delta_j is 1 at spike_indices[j], 0 elsewhere)
    """
    bg = params[0]
    amps = params[1:]
    
    h = np.full(N, bg)
    for j, idx in enumerate(spike_indices):
        h[idx] += amps[j]
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _spike_objective_soft(params, n_spikes, spike_indices, beta):
    """
    Soft-max version of spike objective.
    """
    bg = params[0]
    amps = params[1:]
    
    h = np.full(N, bg)
    for j, idx in enumerate(spike_indices):
        h[idx] += amps[j]
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses DCT spectral representation with soft-max surrogate optimization.
    The DCT basis naturally captures low-frequency structure, and we optimize
    the spectral amplitudes to minimize the peak-to-mean ratio of autoconvolution.
    Also explores spike-based parameterizations for low-dimensional search.
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

    # Family 2: Smoothed block
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 80:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Family 3: Low-frequency DCT basis combinations
    # Represent h = 1 + sum_k a_k * cos(pi*k*(2*i+1)/(2*N))
    # Only use first few DCT modes
    for num_modes in [2, 3, 4]:
        if time.time() > deadline - 70:
            break
        for coeffs in np.random.RandomState(42 + num_modes).randn(50, num_modes):
            h = np.ones(N)
            for k in range(num_modes):
                h += coeffs[k] * np.cos(np.pi * (k+1) * (2*np.arange(N)+1) / (2*N))
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 4: Spike-based parameterizations
    # h = bg + sum_j a_j * delta_{j}
    # The autoconvolution peak is dominated by self-correlation of large elements
    if time.time() < deadline - 60:
        rng_spikes = np.random.RandomState(123)
        for n_spikes in [1, 2, 3, 4]:
            if time.time() > deadline - 60:
                break
            for trial in range(30):
                if time.time() > deadline - 60:
                    break
                # Choose random spike locations
                spike_indices = rng_spikes.choice(N, size=n_spikes, replace=False)
                
                # Initial guess: bg=1, small amplitudes
                params_init = np.concatenate([[1.0], rng_spikes.randn(n_spikes) * 0.1])
                
                try:
                    result = minimize(
                        _spike_objective,
                        params_init,
                        args=(n_spikes, spike_indices),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    bg_opt = result.x[0]
                    amps_opt = result.x[1:]
                    h_opt = np.full(N, bg_opt)
                    for j, idx in enumerate(spike_indices):
                        h_opt[idx] += amps_opt[j]
                    
                    if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                        v_opt = _objective_exact(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                except Exception:
                    continue

    # Phase 2: DCT-domain optimization with soft-max surrogate
    # Convert best_h to DCT coefficients
    c_best = _h_to_dct(best_h)

    # Fix DC component to maintain mean
    c_dc = c_best[0].copy()

    # Two-stage soft-max: low beta first, then high beta
    for beta in [1.0, 5.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline - 25:
            break

        # Objective in DCT domain using soft-max
        def obj_dct(c_ac, beta=beta):
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

        # Try random restarts around best DCT coefficients
        rng = np.random.RandomState(42 + int(beta))
        c_ac_init = c_best[1:].copy()

        for trial in range(4):
            if time.time() > deadline - 25:
                break

            if trial == 0:
                c_ac_trial = c_ac_init.copy()
            else:
                c_ac_trial = c_ac_init + 0.1 * rng.randn(N-1)

            try:
                result = minimize(
                    obj_dct,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
                        c_ac_init = c_best[1:].copy()
            except Exception:
                continue

    # Phase 3: High-precision polish with very high beta in DCT domain
    if time.time() < deadline - 20:
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 15:
                break

            def obj_polish(c_ac, beta=beta):
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
                    obj_polish,
                    c_best[1:].copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )

                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
            except Exception:
                continue

    # Phase 4: Improved coordinate descent with gradient information
    if time.time() < deadline - 10:
        h = best_h.copy()
        
        # Pre-compute for efficiency
        def _compute_conv_and_grad(h):
            """Compute convolution and gradient of the objective."""
            s = np.sum(h)
            if abs(s) < 1e-12:
                return None, None
            conv = _convolve_fft(h)
            max_idx = np.argmax(conv)
            max_val = conv[max_idx]
            v = abs(2.0 * N * max_val / (s * s))
            
            # Gradient with respect to h[i]
            # v = |2N * max(conv) / s^2|
            # conv[k] = sum_i h[i] * h[k-i]
            # d(conv[max_idx])/d(h[j]) = sum_i h[j-i] * delta_{i,k-j} + sum_i h[i] * delta_{k-i,j}
            # = h[max_idx-j] + h[j] (with appropriate bounds)
            
            grad = np.zeros(N)
            for j in range(N):
                # Contribution to conv[max_idx] from h[j]
                term1 = h[max_idx - j] if 0 <= max_idx - j < N else 0
                term2 = h[j] if 0 <= j < N else 0
                dconv_dhj = term1 + term2
                dv_dconv = 2.0 * N / (s * s)
                dv_ds = -2.0 * N * max_val / (s * s * s)
                ds_dhj = 1.0
                grad[j] = dv_dconv * dconv_dhj + dv_ds * ds_dhj
            
            return v, grad
        
        step = 0.05
        for iteration in range(30):
            if time.time() > deadline - 5:
                break
            
            v_curr, grad = _compute_conv_and_grad(h)
            if v_curr is None:
                break
            
            # Line search along negative gradient
            improved = False
            alpha = 0.1
            for ls_iter in range(20):
                h_trial = h - alpha * grad
                v_trial = _objective_exact(h_trial)
                if v_trial < v_curr - 1e-10:
                    h = h_trial
                    best_val = v_trial
                    best_h = h.copy()
                    improved = True
                    break
                alpha *= 0.5
            
            if not improved:
                step *= 0.5
                if step < 0.001:
                    break

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5:
        # Try a few more DCT optimizations with very tight tolerances
        for beta in [2000.0]:
            if time.time() > deadline - 3:
                break
            def obj_final(c_ac, beta=beta):
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
                    v_opt = _objective_exact(h_opt)
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
