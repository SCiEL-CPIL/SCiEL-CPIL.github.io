"""
Combined program: DCT spectral shaping + Golay complementary pair + windowed Zadoff-Chu +
full-dimensional coordinate descent with in-place updates + smooth-max refinement +
sparse atom-based parameterization.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B on smooth max surrogate (multiple restarts).
- Secondary: Golay complementary pair (two chirps with different rates) where the sum of
  individual autocorrelations cancels out off-peak values.
- Tertiary: Windowed Zadoff-Chu sequences with global optimization.
- Fourth: Full-dimensional coordinate descent with in-place updates and smooth logsumexp
  surrogate to escape Fourier subspace restriction, allowing high-frequency corrections.
- Fifth: Sparse atom-based parameterization — model h as sum of shifted/scaled Hamming-window
  atoms, optimizing low-dimensional parameters (amplitudes, integer offsets) via coordinate
  descent and local refinement.
- Refine best candidates with L-BFGS-B on DFT basis and Nelder-Mead polish.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 135.0
_K_DCT = 40
_K_DFT = 20

_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=120.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

def _dct_basis_matrix(K, N):
    """Build DCT-II basis matrix of shape (N, K) for first K modes."""
    n = np.arange(N)
    k = np.arange(K)
    angles = np.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * N)
    basis = np.cos(angles)
    norms = np.sqrt(np.sum(basis ** 2, axis=0))
    norms[norms == 0] = 1.0
    basis = basis / norms
    return basis

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)
_DCT_BASIS_T = _DCT_BASIS.T

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

# Low-frequency DFT basis for refinement
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)

def _dft_smooth_obj(coeffs_base, beta=80.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _golay_complementary_h(params):
    """
    Golay complementary pair: h = base + amp1*cos(quad_phase1) + amp2*cos(quad_phase2)
    Two chirps with different rates can have complementary autocorrelation properties.
    params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
    """
    phase1, rate1, amp1, phase2, rate2, amp2, base = params
    i = _i_arr
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * i * (i + 1) / N + phase2
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h

def _golay_obj(params):
    """Objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _value(h)

def _golay_smooth_obj(params, beta=80.0):
    """Smooth objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu with smooth window to suppress sidelobes."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    i = _i_arr
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    rel_pos = i / N
    
    if win_type < 0.5:
        x = (rel_pos - win_center) / (0.5 * win_width)
        window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    else:
        sigma = win_width / 4.0
        window = np.exp(-0.5 * ((rel_pos - win_center) / sigma) ** 2)
    
    h = base + amp * window * np.cos(ang)
    return h

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=80.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = _i_arr
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

def _project_dft(h):
    """Project h onto DFT basis + base, return (coeffs, base_val)."""
    mean_h = float(np.mean(h))
    coeffs = np.zeros(2 * _K_DFT)
    target = h - mean_h
    for k in range(1, _K_DFT + 1):
        angle = 2.0 * np.pi * k * _n_idx / N
        cos_k = np.cos(angle)
        sin_k = np.sin(angle)
        A = np.column_stack([cos_k, sin_k])
        try:
            sol = np.linalg.lstsq(A, target, rcond=None)[0]
            coeffs[2 * (k - 1)] = sol[0]
            coeffs[2 * (k - 1) + 1] = sol[1]
        except:
            pass
    return coeffs, mean_h

def _full_dim_coord_descent(h0, max_iter=500, step_size=0.01, beta=80.0):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    Uses bisection-based line search per coordinate for faster convergence.
    """
    h = h0.copy()
    best_v = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try a few candidate moves with adaptive step
            best_local = best_v
            best_local_val = old_val
            for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                h[i] = old_val + delta
                v_trial = _smooth_max_value(h, beta)
                if v_trial < best_local:
                    best_local = v_trial
                    best_local_val = old_val + delta
            if best_local < best_v:
                h[i] = best_local_val
                best_v = best_local
                improved = True
            else:
                h[i] = old_val
        if not improved:
            step_size *= 0.7
            if step_size < 1e-7:
                break
    
    return h, best_v

def _hamming_atom(length, center_offset=0):
    """Create a Hamming-windowed atom of given length, centered at center_offset."""
    n = np.arange(length)
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (length - 1))
    atom = np.zeros(N)
    start = center_offset - length // 2
    end = start + length
    # Handle wrap-around
    if start >= 0 and end <= N:
        atom[start:end] = window
    else:
        for idx, w in enumerate(window):
            pos = (start + idx) % N
            atom[pos] += w
    return atom

def _atom_based_h(params, n_atoms, atom_lengths):
    """
    Generate h from sparse atom parameters.
    params: [amp1, off1, amp2, off2, ..., ampK, offK, base]
    where off_i are integer offsets (stored as float for optimizer).
    """
    h = np.zeros(N)
    base = params[-1]
    for i in range(n_atoms):
        amp = params[2 * i]
        off = int(round(params[2 * i + 1]))
        if amp == 0.0:
            continue
        length = atom_lengths[i % len(atom_lengths)]
        h += amp * _hamming_atom(length, off)
    h += base
    return h

def _atom_obj(params, n_atoms, atom_lengths):
    """Objective for atom-based parameters."""
    h = _atom_based_h(params, n_atoms, atom_lengths)
    return _value(h)

def _atom_smooth_obj(params, n_atoms, atom_lengths, beta=80.0):
    """Smooth objective for atom-based parameters."""
    h = _atom_based_h(params, n_atoms, atom_lengths)
    return _smooth_max_value(h, beta)

def _atom_coord_descent(n_atoms, atom_lengths, max_iter=200, beta=80.0, seed=0):
    """
    Coordinate descent on atom parameters (amplitudes and offsets).
    Offsets are integer; we try neighbors. Amplitudes are continuous.
    """
    rng = np.random.default_rng(seed)
    n_params = 2 * n_atoms + 1
    params = np.zeros(n_params)
    for i in range(n_atoms):
        params[2 * i] = rng.uniform(-2.0, 2.0)
        params[2 * i + 1] = rng.uniform(0, N)
    params[-1] = rng.uniform(-1.0, 1.0)
    
    best_v = _atom_smooth_obj(params, n_atoms, atom_lengths, beta)
    amp_step = 0.05
    
    for it in range(max_iter):
        improved = False
        # Optimize amplitudes
        for i in range(n_atoms):
            old_amp = params[2 * i]
            for delta in [amp_step, -amp_step]:
                params[2 * i] = old_amp + delta
                v_trial = _atom_smooth_obj(params, n_atoms, atom_lengths, beta)
                if v_trial < best_v:
                    best_v = v_trial
                    improved = True
                    break
                else:
                    params[2 * i] = old_amp
        # Optimize offsets (integer)
        for i in range(n_atoms):
            old_off = int(round(params[2 * i + 1]))
            for delta in [1, -1, 2, -2]:
                params[2 * i + 1] = old_off + delta
                v_trial = _atom_smooth_obj(params, n_atoms, atom_lengths, beta)
                if v_trial < best_v:
                    best_v = v_trial
                    improved = True
                    break
                else:
                    params[2 * i + 1] = old_off
        # Optimize base
        old_base = params[-1]
        for delta in [amp_step, -amp_step]:
            params[-1] = old_base + delta
            v_trial = _atom_smooth_obj(params, n_atoms, atom_lengths, beta)
            if v_trial < best_v:
                best_v = v_trial
                improved = True
                break
            else:
                params[-1] = old_base
        
        if not improved:
            amp_step *= 0.5
            if amp_step < 1e-6:
                break
    
    h = _atom_based_h(params, n_atoms, atom_lengths)
    return h, best_v

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DCT spectral shaping with L-BFGS-B (multiple restarts with varied frequency bands)
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            # Use different frequency band strategies
            band_strategies = [
                (0, _K_DCT, 4, 12, 1.0),      # all modes, moderate active
                (0, 15, 3, 8, 1.5),           # low freq only
                (5, _K_DCT, 4, 10, 1.0),      # skip DC-ish
                (0, _K_DCT, 2, 6, 2.0),       # sparse, larger amplitude
                (0, 20, 5, 15, 0.8),          # first 20 modes, dense
            ]
            for restart in range(15):
                if not deadline_ok():
                    break
                strategy = band_strategies[restart % len(band_strategies)]
                lo, hi, n_min, n_max, amp_scale = strategy
                coeffs = np.zeros(_K_DCT)
                n_active = int(rng.integers(n_min, min(n_max + 1, hi - lo)))
                active_idx = rng.choice(np.arange(lo, hi), size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, amp_scale, size=n_active)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                              options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 2: Chirp global search
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_chirp = differential_evolution(
                _chirp_obj, bounds_chirp, seed=99, maxiter=30, tol=1e-6,
                popsize=8, init='sobol', polish=False, update_interval=1
            )
            if res_chirp.x is not None:
                h = _chirp_h(res_chirp.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Golay complementary pair global search
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_golay = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=50, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_golay.x is not None:
                h = _golay_complementary_h(res_golay.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_zc = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
                (0.1, 2.0),
                (0.0, 1.0),
                (0.0, 1.0),
            ]
            res_zc = differential_evolution(
                _windowed_zc_obj, bounds_zc, seed=13, maxiter=40, tol=1e-7,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res_zc.x is not None:
                h = _windowed_zadoff_chu_h(res_zc.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: Full-dimensional coordinate descent with in-place updates
    if deadline_ok():
        try:
            h_cd, v_cd = _full_dim_coord_descent(best_h, max_iter=200, step_size=0.03, beta=120.0)
            update_best(h_cd)
        except Exception:
            pass
    
    # Phase 5b: Sparse atom-based parameterization search
    if deadline_ok():
        try:
            atom_configs = [
                (5, [10, 12, 14, 10, 12]),
                (7, [8, 10, 12, 14, 8, 10, 12]),
                (4, [16, 20, 16, 20]),
                (6, [8, 10, 8, 10, 8, 10]),
            ]
            for n_atoms, atom_lengths in atom_configs:
                if not deadline_ok():
                    break
                for seed in range(4):
                    if not deadline_ok():
                        break
                    h_atom, v_atom = _atom_coord_descent(
                        n_atoms, atom_lengths, max_iter=200, beta=120.0, seed=seed
                    )
                    update_best(h_atom)
        except Exception:
            pass
    
    # Phase 6: DFT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            coeffs, base_val = _project_dft(best_h)
            coeffs_base = np.concatenate([coeffs, [base_val]])
            
            res_dft_refine = minimize(
                _dft_smooth_obj,
                coeffs_base,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dft_refine.x is not None:
                c = res_dft_refine.x[:-1]
                b = res_dft_refine.x[-1]
                h = _dft_h(c, b)
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Nelder-Mead polish on DCT basis with smooth objective
    if deadline_ok():
        try:
            dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
            
            res_nm = minimize(
                _dct_smooth_obj,
                dct_coeffs,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 800, 'maxfev': 2000}
            )
            if res_nm.x is not None:
                h = _dct_h(res_nm.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: Final full-dimensional coordinate descent polish with finer steps
    if deadline_ok():
        try:
            h_cd2, v_cd2 = _full_dim_coord_descent(best_h, max_iter=150, step_size=0.008, beta=150.0)
            update_best(h_cd2)
        except Exception:
            pass
    
    # Phase 9: L-BFGS-B refinement on atom parameters from best atom solution
    if deadline_ok():
        try:
            # Try to refine the best atom-based solution with L-BFGS-B on continuous params
            for n_atoms, atom_lengths in [(5, [10, 12, 14, 10, 12]), (6, [8, 10, 8, 10, 8, 10])]:
                if not deadline_ok():
                    break
                n_params = 2 * n_atoms + 1
                # Initialize from a reasonable configuration
                params_init = np.zeros(n_params)
                for i in range(n_atoms):
                    params_init[2 * i] = 0.5 * (1 if i % 2 == 0 else -1)
                    params_init[2 * i + 1] = i * (N // n_atoms)
                params_init[-1] = 0.0
                
                res_atom_lbfgs = minimize(
                    lambda p, na=n_atoms, al=atom_lengths: _atom_smooth_obj(p, na, al, beta=120.0),
                    params_init,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_atom_lbfgs.x is not None:
                    h_atom_ref = _atom_based_h(res_atom_lbfgs.x, n_atoms, atom_lengths)
                    update_best(h_atom_ref)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
