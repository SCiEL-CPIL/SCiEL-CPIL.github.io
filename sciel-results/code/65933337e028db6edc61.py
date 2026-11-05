"""Smooth-max surrogate + Zadoff-Chu chirp + differential evolution."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
DEADLINE = 148.0


def _autocorr_fft(h):
    """Fast full autoconvolution via FFT."""
    n = len(h)
    nfft = 2 ** (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    ac = np.fft.irfft(H * np.conj(H), n=nfft)[:2 * n - 1]
    return ac


def _value(h):
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf
    c = _autocorr_fft(h)
    return abs(2.0 * N * float(np.max(c)) / (S * S))


def _smooth_max_value(h, tau=0.5):
    """Smooth surrogate for max using log-sum-exp."""
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf
    c = _autocorr_fft(h)
    # Softmax-based smooth max
    c_max = np.max(c)
    c_shifted = c - c_max
    log_sum_exp = c_max + np.log(np.sum(np.exp(c_shifted / tau))) * tau
    return abs(2.0 * N * log_sum_exp / (S * S))


def _chirp_h(params):
    """Construct h from chirp parameters: [freq_scale, phase_offset, amplitude_scale]."""
    freq_scale = params[0]
    phase_offset = params[1]
    amp_scale = params[2]
    i = np.arange(N)
    # Zadoff-Chu inspired: quadratic phase (chirp)
    h = amp_scale * np.cos(2.0 * np.pi * freq_scale * i * (i + 1) / (2.0 * N) + phase_offset)
    return h


def _chirp_obj(params):
    """Objective for chirp parameterization."""
    h = _chirp_h(params)
    return _smooth_max_value(h, tau=0.5)


def _dft_basis_h(params):
    """Construct h from DFT basis coefficients (first K complex coefficients)."""
    K = 10
    h = np.zeros(N, dtype=complex)
    for k in range(K):
        h[k] = params[2 * k] + 1j * params[2 * k + 1]
        if k < N - k:
            h[N - k] = params[2 * k] - 1j * params[2 * k + 1]
    h = np.real(np.fft.irfft(h, n=N))
    return h


def _dft_obj(params):
    """Objective for DFT basis parameterization."""
    h = _dft_basis_h(params)
    return _smooth_max_value(h, tau=0.5)


def _zadoff_chu_h():
    """Generate a Zadoff-Chu-like sequence with good autocorrelation properties."""
    # Zadoff-Chu sequence: x[n] = exp(-j*pi*u*n*(n+1)/N) for u coprime to N
    # We use real part to get a real sequence
    u = 1  # Choose u coprime to N
    i = np.arange(N)
    h = np.cos(np.pi * u * i * (i + 1) / N)
    # Normalize to have unit sum
    S = np.sum(h)
    if abs(S) > 1e-12:
        h = h * (N / S)  # Scale so sum = N (like ones)
    return h


def construct():
    t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    # Phase 1: Try Zadoff-Chu inspired chirp
    zc_h = _zadoff_chu_h()
    v_zc = _value(zc_h)
    if v_zc < best_v:
        best_v = v_zc
        best_h = zc_h.copy()
    
    # Phase 2: Differential evolution over chirp parameters
    if time.time() - t0 < DEADLINE - 30:
        try:
            bounds = [
                (0.1, 5.0),      # freq_scale
                (-np.pi, np.pi),  # phase_offset
                (0.5, 3.0),      # amplitude_scale
            ]
            res_de = differential_evolution(
                _chirp_obj,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                updating='deferred',
                workers=1,
            )
            if res_de.x is not None:
                h_de = _chirp_h(res_de.x)
                v_de = _value(h_de)
                if v_de < best_v:
                    best_v = v_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B refinement on the best chirp found
    if time.time() - t0 < DEADLINE - 15:
        try:
            # Get initial chirp params from best_h if it's a chirp, else use reasonable defaults
            if np.allclose(best_h, zc_h):
                # Estimate chirp params from ZC
                freq_scale = 1.0
                phase_offset = 0.0
                amp_scale = N / np.sum(zc_h) if abs(np.sum(zc_h)) > 1e-12 else 1.0
            else:
                freq_scale = 1.0
                phase_offset = 0.0
                amp_scale = 1.0
            
            params_init = np.array([freq_scale, phase_offset, amp_scale])
            
            res_lbfgs = minimize(
                _chirp_obj,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100},
            )
            if res_lbfgs.x is not None:
                h_lbfgs = _chirp_h(res_lbfgs.x)
                v_lbfgs = _value(h_lbfgs)
                if v_lbfgs < best_v:
                    best_v = v_lbfgs
                    best_h = h_lbfgs.copy()
        except Exception:
            pass
    
    # Phase 4: DFT basis optimization
    if time.time() - t0 < DEADLINE - 10:
        try:
            K = 8
            dim = 2 * K
            bounds_dft = [(-5.0, 5.0)] * dim
            
            # Initialize from best_h's DFT
            H = np.fft.rfft(best_h, n=N)
            params_init_dft = np.zeros(dim)
            for k in range(K):
                params_init_dft[2 * k] = H[k].real
                params_init_dft[2 * k + 1] = H[k].imag
            
            res_dft = minimize(
                _dft_obj,
                params_init_dft,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100},
            )
            if res_dft.x is not None:
                h_dft = _dft_basis_h(res_dft.x)
                v_dft = _value(h_dft)
                if v_dft < best_v:
                    best_v = v_dft
                    best_h = h_dft.copy()
        except Exception:
            pass
    
    # Final safety check
    if time.time() - t0 > DEADLINE - 2:
        return np.ones(N)
    
    # Ensure finite and non-zero sum
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
