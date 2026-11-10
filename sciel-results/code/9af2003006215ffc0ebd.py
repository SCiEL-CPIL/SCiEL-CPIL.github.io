# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv

_DEADLINE = time.time() + 600.0
_MARGIN = 15.0

_K = 10
_NTH = 70
_NPHI = 130
_NCOEF = 2 * _K + 1

def _build_basis():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    wq = 2.0 * np.pi * w
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    P = lpmv(np.arange(_K + 1)[None, :], _K, x[:, None])  # shape (NTH, K+1)

    N = np.zeros(_K + 1)
    for m in range(_K + 1):
        ratio = 1.0
        for i in range(_K - m + 1, _K + m + 1):
            ratio /= i
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * ratio)

    B = np.zeros((_NTH, _NPHI, _NCOEF))
    for m in range(_K + 1):
        Nm = N[m]
        if m == 0:
            Y0 = Nm * P[:, 0][:, None]
            B[:, :, _K] = Y0
        else:
            Yc = np.sqrt(2.0) * Nm * P[:, m][:, None] * np.cos(m * phi)[None, :]
            Ys = np.sqrt(2.0) * Nm * P[:, m][:, None] * np.sin(m * phi)[None, :]
            B[:, :, _K + m] = Yc
            B[:, :, _K - m] = Ys

    Bf = B.reshape(_NTH * _NPHI, _NCOEF)
    W = np.repeat(wq, _NPHI) * dphi
    return Bf, W

_BF, _W = _build_basis()

def _compute_R(c):
    f = _BF.dot(c)
    n4 = float(np.sum(_W * f ** 4))
    n2 = float(np.sum(_W * f * f))
    if n2 <= 0.0:
        return np.inf
    return (n4 / n2) ** 0.5

def _compute_R_grad(c):
    f = _BF.dot(c)
    n4 = float(np.sum(_W * f ** 4))
    n2 = float(np.sum(_W * f * f))
    if n2 <= 0.0:
        return np.inf, np.zeros_like(c)
    R = (n4 / n2) ** 0.5
    df4 = _W * f ** 3
    df2 = _W * f
    g = (2.0 * n2 * _BF.T.dot(df4) - n4 * _BF.T.dot(df2) / n2)
    g *= (1.0 / (2.0 * R))
    return R, g

def _line_search(c, R, g, alpha):
    step = -g
    norm_step = np.linalg.norm(step)
    if norm_step < 1e-15:
        return c, R
    step = step / norm_step
    if np.dot(g, step) > 0:
        step = -step
    a = alpha
    R_new = R
    c_new = c
    for _ in range(50):
        cand = c + a * step
        R_cand = _compute_R(cand)
        if R_cand < R_new:
            R_new = R_cand
            c_new = cand
            break
        a *= 0.5
    return c_new, R_new

def _optimize_from(c0, iters, alpha):
    c = c0.copy()
    R = _compute_R(c)
    for it in range(iters):
        if time.time() > _DEADLINE - _MARGIN:
            break
        R, g = _compute_R_grad(c)
        c, R = _line_search(c, R, g, alpha)
    return c, R

def _adam_optimize(c0, iters, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    c = c0.copy()
    m = np.zeros_like(c)
    v = np.zeros_like(c)
    R = _compute_R(c)
    t = 0
    for it in range(iters):
        if time.time() > _DEADLINE - _MARGIN:
            break
        t += 1
        R, g = _compute_R_grad(c)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        c_new = c - step
        R_new = _compute_R(c_new)
        if R_new < R:
            c = c_new
            R = R_new
        else:
            lr *= 0.5
    return c, R

def _newton_refine(c0, iters=50):
    c = c0.copy()
    R = _compute_R(c)
    for it in range(iters):
        if time.time() > _DEADLINE - _MARGIN:
            break
        R, g = _compute_R_grad(c)
        # Hessian approximation via finite differences of gradient
        H = np.zeros((c.size, c.size))
        eps = 1e-5
        for j in range(c.size):
            c_p = c.copy()
            c_p[j] += eps
            _, g_p = _compute_R_grad(c_p)
            c_m = c.copy()
            c_m[j] -= eps
            _, g_m = _compute_R_grad(c_m)
            H[:, j] = (g_p - g_m) / (2 * eps)
        # Solve H * delta = -g
        try:
            delta = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        # Line search on Newton step
        a = 1.0
        R_new = R
        c_new = c
        for _ in range(30):
            cand = c + a * delta
            R_cand = _compute_R(cand)
            if R_cand < R_new:
                R_new = R_cand
                c_new = cand
                break
            a *= 0.5
        c = c_new
        R = R_new
    return c, R

def rudin_problem10() -> np.ndarray:
    rng = np.random.default_rng(42)
    
    # Start with zonal harmonic (m=0) which is known to be good
    best_c = np.zeros(_NCOEF)
    best_c[_K] = 1.0  # m=0 component
    best_R = _compute_R(best_c)
    
    # Refine zonal start
    c, R = _optimize_from(best_c.copy(), 200, 1.0)
    if R < best_R:
        best_c, best_R = c, R
    
    # Newton refinement
    if time.time() < _DEADLINE - _MARGIN:
        c2, R2 = _newton_refine(best_c.copy(), 30)
        if R2 < best_R:
            best_c, best_R = c2, R2
    
    # Random restarts with various strategies
    for restart in range(60):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Different initialization strategies
        strategy = restart % 4
        if strategy == 0:
            # Random
            c0 = rng.normal(size=_NCOEF)
            c0 /= (np.linalg.norm(c0) + 1e-15)
        elif strategy == 1:
            # Zonal + small perturbation
            c0 = np.zeros(_NCOEF)
            c0[_K] = 1.0
            c0 += 0.1 * rng.normal(size=_NCOEF)
        elif strategy == 2:
            # Single mode
            m = int(rng.integers(0, _K + 1))
            c0 = np.zeros(_NCOEF)
            if m == 0:
                c0[_K] = 1.0
            else:
                c0[_K + m] = 1.0
                c0[_K - m] = rng.uniform(-1, 1)
        else:
            # Mixed modes
            c0 = np.zeros(_NCOEF)
            for m in range(_K + 1):
                if rng.random() > 0.5:
                    c0[_K + m] = rng.normal()
                    if m > 0:
                        c0[_K - m] = rng.normal()
            c0 /= (np.linalg.norm(c0) + 1e-15)
        
        # Optimize with gradient descent
        c, R = _optimize_from(c0, 150, 1.0)
        if R < best_R:
            best_R = R
            best_c = c.copy()
            # Polish with Adam
            if time.time() < _DEADLINE - _MARGIN:
                c2, R2 = _adam_optimize(best_c.copy(), 100, lr=0.05)
                if R2 < best_R:
                    best_R = R2
                    best_c = c2.copy()
            # Polish with Newton
            if time.time() < _DEADLINE - _MARGIN:
                c3, R3 = _newton_refine(best_c.copy(), 20)
                if R3 < best_R:
                    best_R = R3
                    best_c = c3.copy()
    
    # Final validation
    if not np.all(np.isfinite(best_c)):
        best_c = np.zeros(_NCOEF)
        best_c[_K] = 1.0
    
    # Ensure valid range
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c *= (1e-1 / max_abs)
    elif max_abs > 1e5:
        best_c *= (1e4 / max_abs)
    
    return best_c.astype(np.float64)
# EVOLVE-BLOCK-END
