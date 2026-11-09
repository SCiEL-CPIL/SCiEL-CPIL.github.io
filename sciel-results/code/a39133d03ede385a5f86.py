# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv
from math import lgamma

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1
_DEADLINE_OFFSET = 590.0
_SAFETY = 20.0

def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        log_ratio = lgamma(_K - am + 1) - lgamma(_K + am + 1)
        fact_ratio = np.exp(log_ratio)
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

def _objective(c):
    """Minimize R^4 = <f^4> / <f^2>^2. Scale-invariant."""
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)

def _gradient(c):
    """Analytic gradient of R^4 = <f^4>/<f^2>^2 w.r.t. c."""
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad

def _zr_to_c(zr):
    """Convert real vector representation to 21-coeff vector."""
    c = np.zeros(_NC)
    c[_K] = zr[0]
    for m in range(1, _K + 1):
        c[_K - m] = zr[2*m - 1]
        c[_K + m] = zr[2*m]
    return c

def _c_to_zr(c):
    """Convert 21-coeff vector to real vector representation."""
    zr = np.zeros(2 * (_K + 1))
    zr[0] = c[_K]
    for m in range(1, _K + 1):
        zr[2*m - 1] = c[_K - m]
        zr[2*m] = c[_K + m]
    return zr

def _objective_zr(zr):
    """Objective in 22-dim real representation."""
    c = _zr_to_c(zr)
    return _objective(c)

def _gradient_zr(zr):
    """Gradient in 22-dim real representation."""
    c = _zr_to_c(zr)
    gc = _gradient(c)
    gzr = np.zeros(2 * (_K + 1))
    gzr[0] = gc[_K]
    for m in range(1, _K + 1):
        gzr[2*m - 1] = gc[_K - m]
        gzr[2*m] = gc[_K + m]
    return gzr

def _projected_cg_momentum(c, deadline, max_iter=3000, momentum=0.9):
    """Projected CG with Nesterov momentum on the unit sphere."""
    c = c.copy()
    val = _objective(c)
    g = _gradient(c)
    g = g - np.dot(g, c) * c  # project onto tangent space
    
    v = np.zeros_like(c)  # velocity for momentum
    d = -g
    d = d - np.dot(d, c) * c
    dn = np.linalg.norm(d)
    if dn < 1e-15:
        return c, val
    d /= dn
    
    for it in range(max_iter):
        if time.time() > deadline - _SAFETY:
            break
        
        # Backtracking line search
        alpha = 0.1
        best_alpha = alpha
        best_val = val
        best_c_new = c.copy()
        for _ in range(40):
            c_new = c + alpha * d
            nrm = np.linalg.norm(c_new)
            if nrm < 1e-14:
                alpha *= 0.5
                continue
            c_new /= nrm
            v_new = _objective(c_new)
            if v_new < best_val:
                best_val = v_new
                best_alpha = alpha
                best_c_new = c_new.copy()
            if v_new >= val * (1.0 - 1e-4 * alpha):
                break
            alpha *= 0.5
            if alpha < 1e-14:
                break
        
        if best_val >= val - 1e-18:
            break
        
        # Update with momentum
        v = momentum * v + (1.0 - momentum) * (-g)
        v = v - np.dot(v, c) * c  # project velocity
        
        c = best_c_new
        val = best_val
        
        g_new = _gradient(c)
        g_new = g_new - np.dot(g_new, c) * c
        
        d_new = v
        d_new = d_new - np.dot(d_new, c) * c
        dn = np.linalg.norm(d_new)
        if dn < 1e-15:
            break
        d = d_new / dn
        g = g_new
    
    return c, val

def _riemannian_lbfgs(c, deadline, max_iter=500):
    """Simple Riemannian L-BFGS on the sphere using scipy's L-BFGS-B with projection."""
    c = c.copy()
    val = _objective(c)
    
    # Store history for L-BFGS
    m = 10  # memory
    s_hist = []
    y_hist = []
    
    g = _gradient(c)
    g = g - np.dot(g, c) * c
    
    for it in range(max_iter):
        if time.time() > deadline - _SAFETY:
            break
        
        # L-BFGS two-loop recursion to compute search direction
        if len(s_hist) == 0:
            d = -g
        else:
            # Two-loop recursion
            rho = [1.0 / np.dot(s, y) for s, y in zip(s_hist, y_hist)]
            q = g.copy()
            alpha = np.zeros(len(s_hist))
            for i in range(len(s_hist) - 1, -1, -1):
                alpha[i] = rho[i] * np.dot(s_hist[i], q)
                q -= alpha[i] * y_hist[i]
            
            # Initial Hessian approximation (identity)
            r = q.copy()
            
            for i in range(len(s_hist)):
                beta = rho[i] * np.dot(y_hist[i], r)
                r += s_hist[i] * (alpha[i] - beta)
            
            d = -r
        
        d = d - np.dot(d, c) * c
        dn = np.linalg.norm(d)
        if dn < 1e-15:
            break
        d /= dn
        
        # Line search
        alpha = 0.1
        best_val = val
        best_c_new = c.copy()
        for _ in range(30):
            c_new = c + alpha * d
            nrm = np.linalg.norm(c_new)
            if nrm < 1e-14:
                alpha *= 0.5
                continue
            c_new /= nrm
            v_new = _objective(c_new)
            if v_new < best_val:
                best_val = v_new
                best_c_new = c_new.copy()
            if v_new >= val * (1.0 - 1e-4 * alpha):
                break
            alpha *= 0.5
            if alpha < 1e-14:
                break
        
        if best_val >= val - 1e-18:
            break
        
        # Update history
        g_new = _gradient(best_c_new)
        g_new = g_new - np.dot(g_new, best_c_new) * best_c_new
        
        s = best_c_new - c
        y = g_new - g
        sy = np.dot(s, y)
        if sy > 1e-16:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
        
        c = best_c_new
        val = best_val
        g = g_new
    
    return c, val

def _optimize_from(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm

    best_c = c.copy()
    best_val = _objective(c)

    # Phase 1: SLSQP
    if time.time() < deadline - 30.0:
        try:
            constraints = [{'type': 'eq', 'fun': lambda x: x @ x - 1.0, 'jac': lambda x: 2.0 * x}]
            res = minimize(_objective, c, jac=_gradient, method="SLSQP", constraints=constraints,
                           options={"maxiter": 200, "ftol": 1e-14, "disp": False})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    best_c = best_c / nrm
    best_val = _objective(best_c)

    # Phase 2: Projected CG with momentum
    if time.time() < deadline - 10.0:
        try:
            c_cg, val_cg = _projected_cg_momentum(best_c, deadline, max_iter=2000)
            if val_cg < best_val:
                best_val = val_cg
                best_c = c_cg.copy()
        except Exception:
            pass

    # Phase 3: Riemannian L-BFGS
    if time.time() < deadline - 10.0:
        try:
            c_lb, val_lb = _riemannian_lbfgs(best_c, deadline, max_iter=300)
            if val_lb < best_val:
                best_val = val_lb
                best_c = c_lb.copy()
        except Exception:
            pass

    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    best_c = best_c / nrm
    return best_c, float(_objective(best_c))

def _make_seeds(rng, n_seeds):
    seeds = []
    for i in range(n_seeds):
        c = np.zeros(_NC)
        if i == 0:
            c[_K] = 1.0
        elif i == 1:
            c[_K + _K] = 1.0
        elif i == 2:
            c[_K - _K] = 1.0
        elif i == 3:
            for m in range(-_K, _K + 1):
                c[m + _K] = np.cos(0.5 * m)
        elif i == 4:
            for m in range(-_K, _K + 1):
                c[m + _K] = (-1) ** m
        elif i == 5:
            c[:] = rng.standard_normal(_NC)
        elif i == 6:
            for m in range(-_K, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 7:
            for m in range(-_K + 1, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 8:
            c[_K] = 1.0
            c[_K + _K] = rng.standard_normal()
            c[_K - _K] = rng.standard_normal()
        elif i == 9:
            c[_K + 5] = rng.standard_normal()
            c[_K - 5] = rng.standard_normal()
        elif i == 10:
            for m in range(0, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = v
        elif i == 11:
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = -v
        elif i == 12:
            for m in range(-3, 4):
                c[m + _K] = rng.standard_normal()
        elif i == 13:
            for m in list(range(-10, -6)) + list(range(7, 11)):
                c[m + _K] = rng.standard_normal()
        elif i == 14:
            c[_K] = 1.0
            c[_K + 5] = 0.5 * rng.standard_normal()
            c[_K - 5] = 0.5 * rng.standard_normal()
            c[_K + 10] = 0.1 * rng.standard_normal()
            c[_K - 10] = 0.1 * rng.standard_normal()
        elif i == 15:
            for m in range(-_K, _K + 1):
                c[m + _K] = rng.standard_normal()
        elif i == 16:
            c[_K] = 2.0
            c[_K + 1] = rng.standard_normal()
            c[_K - 1] = rng.standard_normal()
            c[_K + 2] = 0.3 * rng.standard_normal()
            c[_K - 2] = 0.3 * rng.standard_normal()
        elif i == 17:
            c[_K + 10] = 2.0
            c[_K - 10] = 2.0
            c[_K + 9] = 0.5 * rng.standard_normal()
            c[_K - 9] = 0.5 * rng.standard_normal()
            c[_K] = 0.3 * rng.standard_normal()
        elif i == 18:
            for m in range(-_K, _K + 1):
                c[m + _K] = rng.standard_normal() * (0.7 ** abs(m))
        elif i == 19:
            c[_K] = 1.5
            c[_K + 10] = 1.0 * rng.standard_normal()
            c[_K - 10] = 1.0 * rng.standard_normal()
            for m in range(-4, 5):
                if m != 0:
                    c[m + _K] = 0.2 * rng.standard_normal()
        elif i == 20:
            for m in range(-_K, _K + 1):
                c[m + _K] = rng.standard_normal() / (1.0 + 0.3 * abs(m))
        elif i == 21:
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = v
            c[_K] = rng.standard_normal()
        elif i == 22:
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = -v
        elif i == 23:
            # Concentrate on high |m| with alternating signs
            for m in range(6, _K + 1):
                c[_K + m] = (-1) ** (m % 2)
                c[_K - m] = (-1) ** (m % 2)
        elif i == 24:
            # Only m = ±10
            c[_K + 10] = 1.0
            c[_K - 10] = 1.0
        elif i == 25:
            # m = ±8, ±10
            c[_K + 8] = 1.0
            c[_K - 8] = 1.0
            c[_K + 10] = 0.7
            c[_K - 10] = 0.7
        elif i == 26:
            # Even m: 2, 4, 6, 8, 10
            for m in [2, 4, 6, 8, 10]:
                c[_K + m] = 1.0 / m
                c[_K - m] = 1.0 / m
        elif i == 27:
            # Odd m: 1, 3, 5, 7, 9
            for m in [1, 3, 5, 7, 9]:
                c[_K + m] = 1.0 / m
                c[_K - m] = 1.0 / m
        elif i == 28:
            # Symmetric cos pattern
            for m in range(-_K, _K + 1):
                c[m + _K] = np.cos(np.pi * m / 20.0)
        elif i == 29:
            # Bessel-like decay
            for m in range(-_K, _K + 1):
                c[m + _K] = np.exp(-0.1 * m * m)
        else:
            c[:] = rng.standard_normal(_NC)
            c[_K] += 0.5
        nrm = np.linalg.norm(c)
        if nrm > 1e-14:
            c /= nrm
        seeds.append(c)
    return seeds

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + _DEADLINE_OFFSET
    rng = np.random.default_rng(42)

    best_c = None
    best_val = np.inf

    # Phase A: Structured seeds
    seeds = _make_seeds(rng, 30)
    for i, c0 in enumerate(seeds):
        if time.time() > deadline - _SAFETY:
            break
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Phase B: Random restarts
    n_extra = 0
    while time.time() < deadline - 40.0:
        c0 = rng.standard_normal(_NC)
        nrm = np.linalg.norm(c0)
        if nrm < 1e-14:
            continue
        c0 /= nrm
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()
        n_extra += 1

    # Final high-precision polish
    if best_c is not None and time.time() < deadline - 15.0:
        # Tight SLSQP
        try:
            constraints = [{'type': 'eq', 'fun': lambda x: x @ x - 1.0, 'jac': lambda x: 2.0 * x}]
            res = minimize(_objective, best_c, jac=_gradient, method="SLSQP", constraints=constraints,
                           options={"maxiter": 500, "ftol": 1e-16, "disp": False})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

        # Very tight projected CG
        if time.time() < deadline - 10.0:
            try:
                c_cg, val_cg = _projected_cg_momentum(best_c, deadline, max_iter=5000)
                if val_cg < best_val:
                    best_val = val_cg
                    best_c = c_cg.copy()
            except Exception:
                pass

        # Riemannian L-BFGS final
        if time.time() < deadline - 5.0:
            try:
                c_lb, val_lb = _riemannian_lbfgs(best_c, deadline, max_iter=500)
                if val_lb < best_val:
                    best_val = val_lb
                    best_c = c_lb.copy()
            except Exception:
                pass

    if best_c is None:
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)

    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c
# EVOLVE-BLOCK-END
