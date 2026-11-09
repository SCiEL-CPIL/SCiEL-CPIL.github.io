"""Symbolic regression: nonlinear oscillator acceleration via Fourier-poly basis.

Structural change from RBF local-linearization: the oscillator is driven by t,
so the dominant nonlinearity is periodic in t. We use a basis of Fourier
features (sin/cos at multiple harmonics of t) combined with low-order polynomials
in (x, v) to capture both the periodic forcing and the phase-plane dynamics.

At import time, we search over several basis configurations (number of Fourier
harmonics, which polynomial terms) using a surrogate driven-oscillator model
and select the one with best validation MSE. The final model is linear in the
10 params, so BFGS recovers the least-squares optimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130.0   # explicit wall-clock safety margin (< 150 s kill)

# Training ranges for x and v (from task contract)
_X_MIN, _X_MAX = -0.2056, 0.1749
_V_MIN, _V_MAX = -0.3391, 0.3624


def _build_fourier_poly_basis(t, x, v, n_harm, poly_terms):
    """Build a 10-column feature matrix: Fourier in t + polynomial in (x,v).

    n_harm: number of Fourier harmonics (each gives sin + cos = 2 features)
    poly_terms: list of polynomial feature indices to include in (x,v)
    
    Feature layout:
      - bias (1)
      - sin(k*t), cos(k*t) for k=1..n_harm  (2*n_harm features)
      - polynomial terms from poly_terms
      - pad with zeros if needed to reach 10

    Returns (n, 10) array.
    """
    n = len(t)
    cols = [np.ones(n)]
    
    for k in range(1, n_harm + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    
    # Polynomial terms in (x, v)
    # Term index mapping:
    # 0: x, 1: v, 2: x^2, 3: v^2, 4: x*v, 5: x^3, 6: v^3,
    # 7: x^2*v, 8: x*v^2, 9: x^4, 10: v^4, 11: x^3*v, 12: x*v^3
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    x4 = x2 * x2
    v4 = v2 * v2
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    x3v = x3 * v
    xv3 = x * v3
    
    poly_map = {
        0: x, 1: v, 2: x2, 3: v2, 4: xv,
        5: x3, 6: v3, 7: x2v, 8: xv2,
        9: x4, 10: v4, 11: x3v, 12: xv3,
    }
    
    for idx in poly_terms:
        if idx in poly_map:
            cols.append(poly_map[idx])
    
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
    drive = 0.3 * np.sin(t) + 0.15 * np.cos(t) + 0.08 * np.sin(2*t)
    stiff = 1.5 * x + 3.0 * x * x * x
    damp = 0.4 * v - 0.8 * v * v * v
    couple = 0.5 * x * v + 0.2 * x * x * v
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
    """Search over Fourier-poly configurations and select the best by validation error."""
    t, x, v, y = _build_surrogate()
    idx = np.arange(len(t))
    np.random.default_rng(7).shuffle(idx)
    split = int(0.8 * len(t))
    tr, te = idx[:split], idx[split:]

    best_config = None
    best_score = np.inf

    # Candidate configurations: (n_harm, poly_terms)
    # Total features: 1 (bias) + 2*n_harm + len(poly_terms) <= 10
    configs = []
    
    # Pure Fourier with varying harmonics
    for nh in [1, 2, 3, 4, 5]:
        n_poly = 10 - 1 - 2 * nh
        if n_poly > 0:
            # Pick the most important poly terms
            # Prioritize: x, v, x^2, v^2, xv, x^3, v^3
            all_poly = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            poly = all_poly[:n_poly]
            configs.append((nh, poly))
        elif n_poly == 0:
            configs.append((nh, []))
    
    # Mixed: fewer harmonics, more polynomial
    for nh in [1, 2, 3]:
        n_poly = 10 - 1 - 2 * nh
        if n_poly > 0:
            # Different polynomial selections
            # Option A: linear + quadratic
            poly_a = [0, 1, 2, 3, 4]
            if len(poly_a) <= n_poly:
                configs.append((nh, poly_a[:n_poly]))
            # Option B: emphasize x and v
            poly_b = [0, 1, 0, 1, 2, 3, 4, 5, 6]  # duplicate x,v for emphasis? No, use unique
            poly_b = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            configs.append((nh, poly_b[:n_poly]))
    
    # Remove duplicates
    seen = set()
    unique_configs = []
    for nh, poly in configs:
        key = (nh, tuple(sorted(poly)))
        if key not in seen:
            seen.add(key)
            unique_configs.append((nh, poly))
    configs = unique_configs

    for nh, poly in configs:
        if time.time() > _DEADLINE:
            break
        try:
            F = _build_fourier_poly_basis(t, x, v, nh, poly)
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
            best_config = (nh, poly)

    if best_config is None:
        # Fallback: 3 harmonics + 5 polynomial terms
        best_config = (3, [0, 1, 2, 3, 4])

    return best_config


try:
    _BEST_NHARM, _BEST_POLY = _select_best_config()
except Exception:
    _BEST_NHARM = 3
    _BEST_POLY = [0, 1, 2, 3, 4]

# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction: Fourier-poly basis model.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    F = _build_fourier_poly_basis(t, x, v, _BEST_NHARM, _BEST_POLY)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out
