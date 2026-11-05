"""Chirp-based design: h[i] = cos(phi + a*i + k*i^2), optimized over (phi, a, k)."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
DEADLINE = 145.0  # safety margin inside the 150 s kill
_i = np.arange(N, dtype=float)
_i_centered = _i - N / 2.0


def _autocorr_fft(h):
    """Fast autocorrelation via FFT."""
    n = len(h)
    nfft = 2 ** (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    ac = np.fft.irfft(H * np.conj(H), n=nfft)[:2 * n - 1]
    return ac


def _value(h):
    S = float(np.sum(h))
    if S == 0.0:
        return np.inf
    c = _autocorr_fft(h)
    return abs(2.0 * N * float(np.max(c)) / (S * S))


def _h_from_params(p):
    """h[i] = cos(phi + a*i + k*i^2)"""
    phi, a, k = p
    return np.cos(phi + a * _i + k * _i * _i)


def _obj(p):
    return _value(_h_from_params(p))


def _smooth_max_surrogate(p, beta=50.0):
    """Smooth approximation of max(conv) using log-sum-exp."""
    h = _h_from_params(p)
    S = float(np.sum(h))
    if abs(S) < 1e-10:
        return 1e10
    c = _autocorr_fft(h)
    # Log-sum-exp for smooth max
    c_max = np.max(c)
    c_shifted = c - c_max
    smooth_max = c_max + np.log(np.sum(np.exp(beta * c_shifted))) / beta
    return abs(2.0 * N * smooth_max / (S * S))


def _smooth_max_grad(p, beta=50.0, eps=1e-6):
    """Numerical gradient of smooth max surrogate."""
    grad = np.zeros(len(p))
    f0 = _smooth_max_surrogate(p, beta)
    for i in range(len(p)):
        p_plus = p.copy()
        p_plus[i] += eps
        p_minus = p.copy()
        p_minus[i] -= eps
        fp = _smooth_max_surrogate(p_plus, beta)
        fm = _smooth_max_surrogate(p_minus, beta)
        grad[i] = (fp - fm) / (2 * eps)
    return f0, grad


def _optimize_smooth(p0, beta=50.0, maxiter=200):
    """Minimize smooth surrogate using BFGS."""
    def f_and_g(x):
        return _smooth_max_grad(x, beta)
    
    res = minimize(f_and_g, p0, jac=True, method='BFGS',
                   options={'maxiter': maxiter, 'gtol': 1e-10})
    return res.x


def construct():
    t0 = time.time()
    best_p = (0.0, 0.0, 0.0)
    best_v = _value(_h_from_params(best_p))
    
    # 1) Global search: differential evolution over (phi, a, k)
    bounds = [(-np.pi, np.pi), (-0.1, 0.1), (-0.05, 0.05)]
    try:
        res = differential_evolution(
            _obj,
            bounds,
            seed=0,
            maxiter=50,
            tol=1e-9,
            popsize=15,
            init='sobol',
            polish=False,
        )
        if res.x is not None:
            v = _value(_h_from_params(res.x))
            if v < best_v:
                best_v = v
                best_p = tuple(res.x)
    except Exception:
        pass
    
    # 2) Smooth surrogate optimization from best DE point
    if time.time() - t0 < DEADLINE - 15:
        try:
            p_smooth = _optimize_smooth(best_p, beta=50.0, maxiter=200)
            v = _value(_h_from_params(p_smooth))
            if v < best_v:
                best_v = v
                best_p = tuple(p_smooth)
        except Exception:
            pass
    
    # 3) Local refinement: Nelder-Mead from the best point
    if time.time() - t0 < DEADLINE - 5:
        try:
            res2 = minimize(
                _obj,
                best_p,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500, 'maxfev': 800},
            )
            if res2.x is not None:
                v = _value(_h_from_params(res2.x))
                if v < best_v:
                    best_v = v
                    best_p = tuple(res2.x)
        except Exception:
            pass
    
    # 4) Final high-precision polish
    if time.time() - t0 < DEADLINE - 3:
        try:
            res3 = minimize(
                _obj,
                best_p,
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 300, 'maxfev': 500},
            )
            if res3.x is not None:
                v = _value(_h_from_params(res3.x))
                if v < best_v:
                    best_v = v
                    best_p = tuple(res3.x)
        except Exception:
            pass
    
    # 5) Safety: ensure we return before the deadline
    if time.time() - t0 > DEADLINE - 2:
        return np.ones(N)
    
    return _h_from_params(best_p)
