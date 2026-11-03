"""Optimized C3-prime: minimize max|autoconv| / (sum h)^2 using two-stage optimization.

Strategy:
1. Start from structured initial guesses (constant, alternating, block patterns)
2. Phase 1: BFGS with smooth log-sum-exp surrogate for max|autoconv|
3. Phase 2: Nelder-Mead refinement on the true objective
4. Deadline guard to stay under 150s
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 130.0  # safety margin before 150s kill

def _autoconv(h):
    """Full linear autoconvolution via FFT (fast)."""
    n = len(h)
    size = 2 * n - 1
    H = np.fft.rfft(h, size=size)
    return np.fft.irfft(H * H, size=size)

def _true_obj(h):
    """True objective: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_obj(h, beta):
    """Smooth surrogate: log-sum-exp of |autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    a = np.abs(ac)
    # log-sum-exp trick for stability
    m = np.max(a)
    if m < 1e-30:
        return m / (s * s)
    lse = m + np.log(np.sum(np.exp(a - m))) / beta
    return lse / (s * s)

def _smooth_grad(h, beta):
    """Gradient of smooth surrogate via finite differences (fallback) or analytic."""
    # Use central differences for reliability
    eps = 1e-6
    g = np.zeros_like(h)
    f0 = _smooth_obj(h, beta)
    for i in range(len(h)):
        hp = h.copy(); hp[i] += eps
        hm = h.copy(); hm[i] -= eps
        g[i] = (_smooth_obj(hp, beta) - _smooth_obj(hm, beta)) / (2 * eps)
    return g

def construct():
    t_start = time.time()
    
    # Generate diverse initial guesses
    inits = []
    inits.append(np.ones(N))  # constant
    inits.append(np.ones(N) * np.cos(2 * np.pi * np.arange(N) / N))  # cosine
    inits.append(np.sign(np.sin(2 * np.pi * np.arange(N) / N)) + 0.1 * np.random.randn(N))
    # Block pattern: +1 for first half, -1 for second
    b = np.ones(N); b[N//2:] = -1.0
    inits.append(b + 0.05 * np.random.randn(N))
    # Multi-block
    b2 = np.ones(N)
    for k in range(N):
        b2[k] = np.sign(np.sin(4 * np.pi * k / N))
    inits.append(b2 + 0.05 * np.random.randn(N))
    # Random
    inits.append(np.random.randn(N))
    # Low-frequency
    t = np.linspace(0, 2*np.pi, N)
    inits.append(np.cos(2*t) + 0.3*np.cos(5*t) + 0.1*np.random.randn(N))
    
    best_h = None
    best_val = float('inf')
    
    # Evaluate initial guesses
    for h0 in inits:
        v = _true_obj(h0)
        if v < best_val:
            best_val = v
            best_h = h0.copy()
    
    # Phase 1: BFGS with smooth surrogate from multiple starts
    beta = 50.0
    for h0 in inits[:4]:
        if time.time() > DEADLINE:
            break
        try:
            res = minimize(
                lambda x: _smooth_obj(x, beta),
                h0.copy(),
                jac=lambda x: _smooth_grad(x, beta),
                method='BFGS',
                options={'maxiter': 300, 'gtol': 1e-6}
            )
            h1 = res.x
            v1 = _true_obj(h1)
            if v1 < best_val:
                best_val = v1
                best_h = h1.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on true objective from best so far
    if best_h is not None and time.time() < DEADLINE:
        try:
            res = minimize(
                lambda x: _true_obj(x),
                best_h.copy(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            v2 = _true_obj(res.x)
            if v2 < best_val:
                best_val = v2
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: If time remains, do another round with higher beta and more starts
    if time.time() < DEADLINE - 10:
        beta2 = 100.0
        for h0 in inits[:3]:
            if time.time() > DEADLINE - 5:
                break
            # Perturb best solution
            h_pert = best_h + 0.1 * np.random.randn(N)
            try:
                res = minimize(
                    lambda x: _smooth_obj(x, beta2),
                    h_pert,
                    jac=lambda x: _smooth_grad(x, beta2),
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-5}
                )
                h3 = res.x
                v3 = _true_obj(h3)
                if v3 < best_val:
                    best_val = v3
                    best_h = h3.copy()
            except Exception:
                pass
        
        # Final Nelder-Mead polish
        if time.time() < DEADLINE - 3:
            try:
                res = minimize(
                    lambda x: _true_obj(x),
                    best_h.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                v4 = _true_obj(res.x)
                if v4 < best_val:
                    best_val = v4
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Final safety: ensure we return something valid
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
