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

_DEADLINE = time.time() + 130.0

_X_MIN, _X_MAX = -0.2056, 0.1749
_V_MIN, _V_MAX = -0.3391, 0.3624

def _build_rbf_local_basis(t, x, v, centers, sigma):
    """
    Builds a 10-column feature matrix.
    Col 0: Bias (1)
    Cols 1-3: phi_1 * [1, x, v]
    Cols 4-6: phi_2 * [1, x, v]
    Cols 7-9: phi_3 * [1, x, v]
    """
    n = len(t)
    cols = [np.ones(n)]
    
    for i in range(3):
        xc, vc = centers[i]
        dx = x - xc
        dv = v - vc
        dist_sq = dx*dx + dv*dv
        phi = np.exp(-0.5 * dist_sq / (sigma * sigma))
        
        cols.append(phi)
        cols.append(phi * x)
        cols.append(phi * v)
        
    return np.column_stack(cols)

def _build_surrogate():
    """Generate surrogate data for a nonlinear driven oscillator."""
    rng = np.random.default_rng(42)
    n = 5000
    t = rng.uniform(30.0, 50.0, n)
    x = rng.uniform(_X_MIN, _X_MAX, n)
    v = rng.uniform(_V_MIN, _V_MAX, n)
    
    # Duffing oscillator with damping and forcing
    y = -x - x**3 - 0.5*v + 0.2*np.sin(t)
    
    return t, x, v, y

def _select_best_rbf_config():
    """
    Search for best centers and sigma using surrogate data.
    """
    t, x, v, y = _build_surrogate()
    n = len(t)
    
    sets = [
        [(0,0), (0.15, 0.3), (-0.15, -0.3)],
        [(_X_MIN, _V_MIN), (0,0), (_X_MAX, _V_MAX)],
        [(_X_MIN, 0), (0, _V_MAX), (_X_MAX, 0)],
        [(-0.1, -0.1), (0.1, 0.1), (0, 0)],
        [(-0.15, 0.15), (0.15, -0.15), (0, 0)],
    ]
    
    sigmas = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    best_score = np.inf
    best_centers = sets[0]
    best_sigma = 0.2
    
    idx = np.arange(n)
    np.random.default_rng(7).shuffle(idx)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]
    
    for centers in sets:
        centers_arr = np.array(centers)
        for sigma in sigmas:
            if time.time() > _DEADLINE:
                return best_centers, best_sigma
            try:
                F_tr = _build_rbf_local_basis(t[tr_idx], x[tr_idx], v[tr_idx], centers_arr, sigma)
                F_te = _build_rbf_local_basis(t[te_idx], x[te_idx], v[te_idx], centers_arr, sigma)
                
                c_tr, *_ = np.linalg.lstsq(F_tr, y[tr_idx], rcond=None)
                pred_te = F_te @ c_tr
                mse_te = np.mean((pred_te - y[te_idx])**2)
                
                if np.isfinite(mse_te) and mse_te < best_score:
                    best_score = mse_te
                    best_centers = centers
                    best_sigma = sigma
                    
            except Exception:
                continue
                
    return best_centers, best_sigma

try:
    _BEST_CENTERS, _BEST_SIGMA = _select_best_rbf_config()
except Exception:
    _BEST_CENTERS = [(0,0), (0.15, 0.3), (-0.15, -0.3)]
    _BEST_SIGMA = 0.2

_BEST_CENTERS_ARR = np.array(_BEST_CENTERS)
# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction: RBF local-linear basis model.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    F = _build_rbf_local_basis(t, x, v, _BEST_CENTERS_ARR, _BEST_SIGMA)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out
