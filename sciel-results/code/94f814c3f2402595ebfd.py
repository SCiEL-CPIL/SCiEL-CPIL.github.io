# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv

# Precomputation: build the quadrature and the spherical-harmonic matrix Y.
# Quadrature: 70-point Gauss-Legendre in u = cos(theta), 130 equally spaced phi.
_NU = 70
_NPHI = 130
_K = 10

def _build_quadrature():
    # Gauss-Legendre nodes and weights on [-1, 1]
    u, wu = np.polynomial.legendre.leggauss(_NU)
    # phi grid, equally spaced
    phi = 2.0 * np.pi * np.arange(_NPHI) / _NPHI
    # Surface measure dOmega = dphi * du ; total mass 4*pi.
    # Weight for each (u, phi) point: wu_i * (2*pi/_NPHI)
    wphi = 2.0 * np.pi / _NPHI
    # Build full weight grid (NU, NPHI)
    W = np.outer(wu, wphi)  # shape (NU, NPHI)
    # Verify total mass
    assert np.isclose(W.sum(), 4.0 * np.pi, rtol=1e-10), W.sum()
    return u, phi, W

def _spherical_harmonics(u, phi):
    """
    Return real orthonormal degree-K spherical harmonics Y_m for m=-K..K
    evaluated on the grid (NU, NPHI).
    """
    k = _K
    n = 2 * k + 1
    Y = np.zeros((u.shape[0], phi.shape[0], n), dtype=np.float64)
    import math
    log_fact = np.array([math.lgamma(i + 1) for i in range(2 * k + 1)])
    for m in range(-k, k + 1):
        am = abs(m)
        logN = 0.5 * (math.log(21.0) - math.log(4.0 * np.pi) + log_fact[k - am] - log_fact[k + am])
        N = math.exp(logN)
        P = lpmv(am, k, u)
        idx = m + k
        if m == 0:
            Y[:, :, idx] = N * P
        elif m > 0:
            Y[:, :, idx] = math.sqrt(2.0) * N * P * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = math.sqrt(2.0) * N * P * np.sin(am * phi)[None, :]
    return Y

# Build global objects once at import time
_U, _PHI, _W = _build_quadrature()
_Y = _spherical_harmonics(_U, _PHI)
_Yflat = _Y.reshape(-1, 2 * _K + 1)
_Wflat = _W.reshape(-1)
_N = _Yflat.shape[0]

def _f_at(c):
    return _Yflat @ c

def _compute_val_grad(c):
    """
    Compute ||f||_4^4 and its gradient w.r.t c.
    val = sum w_p f_p^4
    grad = 4 Y^T (w * f^3)
    """
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    val = np.dot(_Wflat, f3 * f)
    grad = 4.0 * (_Yflat.T @ (_Wflat * f3))
    return val, grad

def _ratio(c):
    """Compute R = ||f||_4 / ||f||_2."""
    f = _Yflat @ c
    n2 = np.dot(c, c)
    n4 = np.dot(_Wflat, f * f * f * f)
    if n2 <= 0:
        return np.inf
    return (n4 ** 0.25) / (n2 ** 0.5)

def _validate(c):
    if c is None or c.shape != (21,):
        return False
    if not np.all(np.isfinite(c)):
        return False
    mx = np.max(np.abs(c))
    if mx < 1e-3 or mx > 1e5:
        return False
    return True

def _optimize_local(c0, max_iter=500, tol=1e-10):
    """
    Gradient descent on the sphere with backtracking line search.
    Minimizes ||f||_4^4 subject to ||c||=1.
    """
    c = c0.copy()
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-10:
        return None
    c = c / norm_c
    
    val, grad = _compute_val_grad(c)
    
    for i in range(max_iter):
        # Project gradient to tangent space
        proj = np.dot(grad, c)
        grad_t = grad - proj * c
        
        grad_norm_sq = np.dot(grad_t, grad_t)
        if grad_norm_sq < tol:
            break
        
        # Backtracking line search
        alpha = 0.5
        c_new = None
        val_new = None
        for _ in range(20):
            c_trial = c - alpha * grad_t
            norm_trial = np.linalg.norm(c_trial)
            if norm_trial < 1e-10:
                alpha *= 0.5
                continue
            c_trial = c_trial / norm_trial
            
            f_trial = _Yflat @ c_trial
            val_trial = np.dot(_Wflat, f_trial * f_trial * f_trial * f_trial)
            
            # Armijo condition: val_new <= val - c1 * alpha * ||grad_t||^2
            if val_trial <= val - 0.01 * alpha * grad_norm_sq:
                c_new = c_trial
                val_new = val_trial
                break
            alpha *= 0.5
        
        if c_new is None:
            # If line search fails, try a small fixed step
            c_new = c - 0.01 * grad_t
            norm_trial = np.linalg.norm(c_new)
            if norm_trial < 1e-10:
                break
            c_new = c_new / norm_trial
            f_trial = _Yflat @ c_new
            val_new = np.dot(_Wflat, f_trial * f_trial * f_trial * f_trial)
            
        c = c_new
        val = val_new
        grad, _ = _compute_val_grad(c) # Update grad for next iteration
        
    return c

def rudin_problem10() -> np.ndarray:
    deadline = time.time() + 600.0
    safety = 10.0
    rng = np.random.default_rng(42)
    
    best_c = None
    best_val = np.inf # Minimizing L4^4 is equivalent to minimizing R for fixed L2=1
    
    def _consider(c):
        nonlocal best_c, best_val
        if not _validate(c):
            return
        val, _ = _compute_val_grad(c)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_c = c.copy()

    # Seed
    seed = np.ones(2 * _K + 1, dtype=np.float64)
    seed /= np.linalg.norm(seed)
    _consider(seed)
    
    # Structured starts
    # 1. Pure m=0
    c0 = np.zeros(21); c0[10] = 1.0; _consider(c0)
    # 2. Pure m=10
    c0 = np.zeros(21); c0[20] = 1.0; _consider(c0)
    # 3. Pure m=-10
    c0 = np.zeros(21); c0[0] = 1.0; _consider(c0)
    
    n_restarts = 200
    for ridx in range(n_restarts):
        if time.time() > deadline - safety:
            break
        
        if ridx == 0:
            c_start = seed.copy()
        else:
            c_start = rng.standard_normal(21)
            c_start /= np.linalg.norm(c_start)
        
        try:
            c_opt = _optimize_local(c_start, max_iter=300, tol=1e-12)
            if c_opt is not None:
                _consider(c_opt)
        except Exception:
            pass
            
    if best_c is None:
        best_c = seed
        
    return best_c

# EVOLVE-BLOCK-END
