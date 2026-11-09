# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70
_NC = 2 * _K + 1
_NPTS = _NTH * _NPHI


def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W


_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)


def _ratio_from_coeffs(c):
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 0.0:
        return np.inf
    l4sq = float(np.dot(_Wflat, f2 * f2))
    if l4sq <= 0.0:
        return 1.0
    return np.sqrt(l4sq) / np.sqrt(l2sq)


def _objective_and_gradient(c):
    """
    Objective: J(c) = - (int f^4) / (int f^2)^2
    Gradient: 
    Let I2 = int f^2, I4 = int f^4
    J = - I4 * I2^-2
    dJ/dc = - ( dI4/dc * I2^-2 + I4 * d(I2^-2)/dc )
    d(I2^-2)/dc = -2 * I2^-3 * dI2/dc
    dI2/dc = 2 * Y^T (w * f)
    dI4/dc = 4 * Y^T (w * f^3)
    dJ/dc = - [ 4 * Y^T (w * f^3) * I2^-2 - 2 * I4 * I2^-3 * 2 * Y^T (w * f) ]
    dJ/dc = - I2^-2 * [ 4 * Y^T (w * f^3) - 2 * (I4/I2) * Y^T (w * f) ]
    """
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30, np.zeros_like(c)
    
    l4sq = float(np.dot(_Wflat, f2 * f2))
    
    val = -l4sq / (l2sq * l2sq)
    
    # Gradient terms
    # Term 1: Y^T (w * f^3)
    # Term 2: Y^T (w * f)
    g1 = _Yflat.T @ (_Wflat * f3)
    g2 = _Yflat.T @ (_Wflat * f)
    
    grad = - (1.0 / (l2sq * l2sq)) * (4.0 * g1 - 2.0 * (l4sq / l2sq) * g2)
    
    return val, grad


def _optimize_from(c0, deadline):
    c = np.asarray(c0, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm

    def fun(x):
        v, _ = _objective_and_gradient(x)
        return v

    def jac(x):
        _, g = _objective_and_gradient(x)
        return g

    try:
        # Use L-BFGS-B with analytical gradient
        res = minimize(fun, c, jac=jac, method="L-BFGS-B",
                       options={"maxiter": 200, "maxfun": 1000,
                                "ftol": 1e-15, "gtol": 1e-10})
    except Exception:
        return None
    
    c_opt = np.asarray(res.x, dtype=np.float64)
    nrm_opt = np.linalg.norm(c_opt)
    if nrm_opt < 1e-14:
        return None
    c_opt = c_opt / nrm_opt
    
    val_opt, _ = _objective_and_gradient(c_opt)
    return c_opt, val_opt


def _make_seeds(rng):
    seeds = []
    # 1. Random
    for _ in range(20):
        seeds.append(rng.standard_normal(_NC))
        
    # 2. Low order dominated (m=0, +/-1, +/-2)
    for _ in range(10):
        c = np.zeros(_NC)
        # indices: 0..20 correspond to m=-10..10
        # m=0 is index 10
        # m=+/-1 are 9, 11
        # m=+/-2 are 8, 12
        c[10] = 1.0
        c[9] = rng.standard_normal() * 0.5
        c[11] = rng.standard_normal() * 0.5
        c[8] = rng.standard_normal() * 0.5
        c[12] = rng.standard_normal() * 0.5
        seeds.append(c)
        
    # 3. High order dominated (m=+/-10, +/-9)
    for _ in range(10):
        c = np.zeros(_NC)
        c[0] = 1.0 # m=-10
        c[20] = 1.0 # m=10
        c[1] = rng.standard_normal() * 0.5 # m=-9
        c[19] = rng.standard_normal() * 0.5 # m=9
        seeds.append(c)
        
    # 4. Symmetric even m only
    for _ in range(10):
        c = np.zeros(_NC)
        for m in range(0, _K+1, 2):
            idx_pos = m + _K
            idx_neg = -m + _K
            val = rng.standard_normal()
            c[idx_pos] = val
            c[idx_neg] = val
        seeds.append(c)
        
    return seeds


def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 590.0  # Safety margin
    rng = np.random.default_rng(2024)
    
    seeds = _make_seeds(rng)
    
    best_c = None
    best_val = np.inf
    
    for i, c0 in enumerate(seeds):
        if time.time() > deadline - 5.0:
            break
        
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
            
        if out is None:
            continue
            
        c_opt, val_opt = out
        
        if val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = np.ones(_NC, dtype=np.float64)

    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(_NC, dtype=np.float64)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c


# EVOLVE-BLOCK-END
