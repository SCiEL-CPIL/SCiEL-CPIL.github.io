"""Symbolic regression: nonlinear oscillator acceleration via Chebyshev-FFT basis.

Combines the structural idea of a physics-informed sparse basis (from inspiration 1)
with a refinement strategy using Chebyshev polynomials in normalized state variables
and Fourier features in time (from the current program's Fourier approach, but
replacing monomial polynomials with better-conditioned Chebyshev polynomials).

The basis is linear in 10 parameters, so BFGS recovers the least-squares optimum.
At import time, we select the best combination of Fourier harmonics and Chebyshev
degrees using a surrogate driven-oscillator model.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130.0   # explicit wall-clock safety margin (< 150 s kill)

# Training ranges for x and v (from task contract)
_X_MIN, _X_MAX = -0.2056, 0.1749
_V_MIN, _V_MAX = -0.3391, 0.3624


def _chebyshev_1d(u, n_terms):
    """Compute Chebyshev polynomials T_0(u), T_1(u), ..., T_{n_terms-1}(u).
    
    Returns an array of shape (n, n_terms).
    Uses the recurrence: T_0 = 1, T_1 = u, T_k = 2*u*T_{k-1} - T_{k-2}.
    """
    n = len(u)
    T = np.zeros((n, n_terms))
    T[:, 0] = 1.0
    if n_terms > 1:
        T[:, 1] = u
    for k in range(2, n_terms):
        T[:, k] = 2.0 * u * T[:, k-1] - T[:, k-2]
    return T


def _build_chebyshev_fft_basis(t, x, v, n_harm, n_cheb_x, n_cheb_v, n_cross):
    """Build a 10-column feature matrix: Fourier in t + Chebyshev in (x, v) + cross terms.
    
    n_harm: number of Fourier harmonics (each gives sin + cos = 2 features)
    n_cheb_x: number of Chebyshev terms in x
    n_cheb_v: number of Chebyshev terms in v
    n_cross: number of cross terms (products of Chebyshev in x and v)
    
    Feature layout:
      - bias (1)
      - sin(k*t), cos(k*t) for k=1..n_harm  (2*n_harm features)
      - Chebyshev terms in x (excluding T_0 since bias covers it)
      - Chebyshev terms in v (excluding T_0 since bias covers it)
      - Cross terms: T_i(x) * T_j(v) for selected (i,j)
      - pad with zeros if needed to reach 10
    
    Returns (n, 10) array.
    """
    n = len(t)
    cols = [np.ones(n)]
    
    for k in range(1, n_harm + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    
    # Normalize x and v to [-1, 1] for Chebyshev
    x_norm = 2.0 * (x - _X_MIN) / (_X_MAX - _X_MIN) - 1.0
    v_norm = 2.0 * (v - _V_MIN) / (_V_MAX - _V_MIN) - 1.0
    
    # Chebyshev in x (skip T_0 since bias is already included)
    if n_cheb_x > 0:
        Tx = _chebyshev_1d(x_norm, n_cheb_x + 1)
        for i in range(1, n_cheb_x + 1):
            cols.append(Tx[:, i])
    
    # Chebyshev in v (skip T_0 since bias is already included)
    if n_cheb_v > 0:
        Tv = _chebyshev_1d(v_norm, n_cheb_v + 1)
        for j in range(1, n_cheb_v + 1):
            cols.append(Tv[:, j])
    
    # Cross terms: T_i(x) * T_j(v) for selected pairs
    if n_cross > 0:
        Tx_full = _chebyshev_1d(x_norm, max(n_cheb_x, 1) + 1)
        Tv_full = _chebyshev_1d(v_norm, max(n_cheb_v, 1) + 1)
        # Select cross terms: (1,1), (2,1), (1,2), (2,2), etc.
        cross_pairs = [(1,1), (2,1), (1,2), (2,2), (3,1), (1,3), (3,2), (2,3), (3,3)]
        for i, j in cross_pairs[:n_cross]:
            if i < Tx_full.shape[1] and j < Tv_full.shape[1]:
                cols.append(Tx_full[:, i] * Tv_full[:, j])
    
    # Ensure exactly 10 columns
    while len(cols) < 10:
        cols.append(np.zeros(n))
    cols = cols[:10]
    
    return np.column_stack(cols)


def _build_surrogate():
    """Synthesize plausible nonlinear-oscillator acceleration data for basis selection."""
    rng = np.random.default_rng(1234)
    n = 20000
    t = rng.uniform(30.0, 50.0, n)
    x = rng.uniform(_X_MIN, _X_MAX, n)
    v = rng.uniform(_V_MIN, _V_MAX, n)

    # Driven Duffing oscillator with nonlinear damping and coupling
    drive = 0.3 * np.sin(t) + 0.15 * np.cos(t) + 0.08 * np.sin(2*t) + 0.05 * np.cos(2*t)
    stiff = 1.5 * x + 3.0 * x * x * x + 0.5 * x * x * x * x
    damp = 0.4 * v - 0.8 * v * v * v + 0.3 * v * v
    couple = 0.5 * x * v + 0.2 * x * x * v + 0.1 * x * v * v
    y = -stiff - damp + drive + couple
    return t, x, v, y


def _fit_linear(F, y):
    """Least-squares fit; return (coefficients, residual std)."""
    try:
        c, *_ = np.linalg.lstsq(F, y, rcond=None)
        r = y - F @ c
        return c, float(np.sqrt(np.mean(r * r)))
    except Exception:
        return None, np.inf


def _select_best_config():
    """Search over Chebyshev-FFT configurations and select the best by validation error."""
    t, x, v, y = _build_surrogate()
    idx = np.arange(len(t))
    rng = np.random.default_rng(7)
    rng.shuffle(idx)
    split = int(0.8 * len(t))
    tr, te = idx[:split], idx[split:]

    best_config = None
    best_score = np.inf

    # Candidate configurations: (n_harm, n_cheb_x, n_cheb_v, n_cross)
    # Total features: 1 (bias) + 2*n_harm + n_cheb_x + n_cheb_v + n_cross <= 10
    configs = []
    
    # Vary harmonics from 1 to 4
    for nh in [1, 2, 3, 4]:
        remaining = 10 - 1 - 2 * nh
        if remaining <= 0:
            configs.append((nh, 0, 0, 0))
            continue
        
        # Distribute remaining features among Chebyshev x, v, and cross
        # Try several distributions
        for nx in range(0, min(remaining, 4) + 1):
            rem_after_x = remaining - nx
            if rem_after_x < 0:
                continue
            for nv in range(0, min(rem_after_x, 4) + 1):
                rem_after_v = rem_after_x - nv
                if rem_after_v < 0:
                    continue
                for nc in range(0, min(rem_after_v, 3) + 1):
                    total = 1 + 2*nh + nx + nv + nc
                    if total <= 10:
                        configs.append((nh, nx, nv, nc))
    
    # Remove duplicates
    seen = set()
    unique_configs = []
    for cfg in configs:
        if cfg not in seen:
            seen.add(cfg)
            unique_configs.append(cfg)
    configs = unique_configs

    for nh, nx, nv, nc in configs:
        if time.time() > _DEADLINE:
            break
        try:
            F = _build_chebyshev_fft_basis(t, x, v, nh, nx, nv, nc)
        except Exception:
            continue
        if not (F.shape == (len(t), 10) and np.all(np.isfinite(F))):
            continue
        _, err_tr = _fit_linear(F[tr], y[tr])
        if not np.isfinite(err_tr):
            continue
        _, err_te = _fit_linear(F[te], y[te])
        if not np.isfinite(err_te):
            continue
        gap = err_te - err_tr
        score = err_te + 0.1 * max(0.0, gap)
        if score < best_score:
            best_score = score
            best_config = (nh, nx, nv, nc)

    if best_config is None:
        # Fallback: 2 harmonics, 2 Chebyshev in x, 2 in v, 2 cross
        best_config = (2, 2, 2, 2)

    return best_config


try:
    _BEST_NHARM, _BEST_NX, _BEST_NV, _BEST_NC = _select_best_config()
except Exception:
    _BEST_NHARM = 2
    _BEST_NX = 2
    _BEST_NV = 2
    _BEST_NC = 2

# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction: Chebyshev-FFT basis model.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    F = _build_chebyshev_fft_basis(t, x, v, _BEST_NHARM, _BEST_NX, _BEST_NV, _BEST_NC)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out
