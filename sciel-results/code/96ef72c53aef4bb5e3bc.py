import numpy as np
from scipy.optimize import minimize
import time

N = 600

def _score_from_heights(h):
    """Compute normalized max autoconvolution. Constant should give ~1.0."""
    # Linear autoconvolution via FFT
    n = 2 * N
    f = np.zeros(n, dtype=np.complex128)
    f[:N] = h
    F = np.fft.fft(f)
    A = np.fft.ifft(F * np.conj(F)).real
    A = A[:2 * N - 1]  # linear autoconv length 2N-1
    peak = np.max(A)
    total = np.sum(A)  # = (sum h)^2 * dw, but ratio peak/total is scale-invariant
    if total <= 0:
        return 1.0
    return peak / total * N  # normalize so constant gives ~1.0

def _compute_score_fast(h):
    n = 2 * N
    f = np.zeros(n, dtype=np.complex128)
    f[:N] = h
    F = np.fft.fft(f)
    A = np.fft.ifft(F * np.conj(F)).real
    A = A[:2 * N - 1]
    peak = np.max(A)
    total = np.sum(A)
    if total <= 0:
        return 1.0
    return peak / total * N

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Start with constant
    h0 = np.ones(N)
    best_h = h0.copy()
    best_score = _score_from_heights(h0)

    # Try a few parametric initializations and optimize
    # Use log-parametrization for nonnegativity: h = exp(x), x in R
    # Optimize x to minimize score

    def objective(x):
        h = np.exp(x)
        return _compute_score_fast(h)

    def gradient(x):
        h = np.exp(x)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(N):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            grad[i] = (objective(np.log(np.maximum(np.exp(xp), 1e-30))) - 
                       objective(np.log(np.maximum(np.exp(xm), 1e-30)))) / (2 * eps)
        return grad

    # Better: use finite differences on h directly with bounds, via log transform
    # L-BFGS-B on x = log(h), h >= small_positive

    def obj_x(x):
        h = np.exp(x)
        return _compute_score_fast(h)

    inits = []
    # Constant
    inits.append(np.zeros(N))
    # Bell shape
    x = np.linspace(-1, 1, N)
    inits.append(1.0 - x**2)
    # Cosine bump
    inits.append(1.0 + np.cos(2 * np.pi * x / 2.0))
    # Triangle
    inits.append(np.abs(x))
    # Random smooth
    for _ in range(5):
        r = np.random.randn(N)
        # Smooth
        r = np.convolve(r, np.ones(20)/20, mode='same')
        r = np.maximum(r, 0.1)
        inits.append(np.log(r))

    best_score = _score_from_heights(np.ones(N))
    best_h = np.ones(N)

    for init in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(obj_x, init, method='L-BFGS-B', 
                          options={'maxiter': 200, 'gtol': 1e-8})
            h_res = np.exp(res.x)
            s_res = _compute_score_fast(h_res)
            if s_res < best_score:
                best_score = s_res
                best_h = h_res.copy()
        except Exception:
            continue

    # Local refinement: coordinate descent on the best
    h = best_h.copy()
    s = best_score
    for _ in range(3):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for factor in [0.5, 0.8, 1.2, 1.5]:
                h_new = h.copy()
                h_new[i] = h[i] * factor
                s_new = _compute_score_fast(h_new)
                if s_new < s:
                    h = h_new
                    s = s_new
                    improved = True
        if not improved:
            break

    if s < best_score:
        best_score = s
        best_h = h.copy()

    return list(best_h)
