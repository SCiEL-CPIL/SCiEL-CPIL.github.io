import numpy as np
from scipy.optimize import minimize
import time

N = 400
DEADLINE = 130  # seconds, safety margin from 150s

def _autoconv(h):
    """Compute linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    # Use next power of 2 for FFT efficiency
    fsize = 1
    while fsize < size:
        fsize <<= 1
    F = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(F * np.conj(F), fsize)[:size]
    return ac

def _score(h):
    ac = _autoconv(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)

def _smooth_obj(h, beta=50.0):
    """Smooth approximation of max|ac| / s^2."""
    ac = _autoconv(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    # Smooth max via log-sum-exp
    x = beta * np.abs(ac)
    x_max = np.max(x)
    logsumexp = x_max + np.log(np.sum(np.exp(x - x_max)))
    smooth_max = logsumexp / beta
    return smooth_max / (s * s)

def _smooth_grad(h, beta=50.0):
    """Gradient of smooth objective."""
    ac = _autoconv(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h), 1e10
    
    x = beta * np.abs(ac)
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    weights = exp_x / np.sum(exp_x)  # softmax weights
    
    # Gradient of smooth_max w.r.t. ac
    # d/dac_i [logsumexp(beta*|ac|)/beta] = weights[i] * sign(ac[i])
    dsm_dac = weights * np.sign(ac)
    
    # Gradient of ac w.r.t. h
    # ac[k] = sum_i h[i] * h[i+k]
    # d(ac[k])/d(h[j]) = h[j+k] (if j+k < N) + h[j-k] (if j-k >= 0)
    # For all k, sum_k dsm_dac[k] * d(ac[k])/d(h[j])
    
    n = len(h)
    grad = np.zeros(n)
    for k in range(len(ac)):
        w = dsm_dac[k]
        if w == 0:
            continue
        # Contribution to grad[j] from ac[k]
        # ac[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
        # d(ac[k])/d(h[j]) = h[j+k] if j < N-k, else 0 (from first term)
        #                    + h[j-k] if j >= k, else 0 (from second term, when i=j-k)
        if k == 0:
            # ac[0] = sum h[i]^2, d/dh[j] = 2*h[j]
            grad[j] += 2 * w * h[j]
        else:
            # For j in [0, N-k-1]: h[j+k]
            # For j in [k, N-1]: h[j-k]
            valid1 = slice(0, n - k)
            valid2 = slice(k, n)
            grad[valid1] += w * h[valid1 + k]  # h[j+k] for j in [0, N-k-1]
            grad[valid2] += w * h[valid2 - k]  # h[j-k] for j in [k, N-1]
    
    # Now we have d(smooth_max)/d(h) = grad
    # d(obj)/d(h) = d(smooth_max)/d(h) * 1/s^2 + smooth_max * d(1/s^2)/d(h)
    # d(1/s^2)/d(h[j]) = -2/s^3
    obj = _smooth_obj(h, beta)
    full_grad = grad / (s * s) + obj * (-2.0 / (s ** 3)) * np.ones(n)
    
    return full_grad, obj

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Diverse initial points
    inits = [
        np.ones(N),
        np.random.choice([-1, 1], N),
        np.sign(np.random.randn(N)),
        np.ones(N) * np.random.choice([-1, 1], N),
        # Sparse pattern
        np.zeros(N) + 0.01,
        # Alternating
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
    ]
    
    for init in inits:
        if time.time() - t0 > DEADLINE:
            break
        
        h0 = init.copy()
        try:
            res = minimize(
                _smooth_obj, h0,
                jac=lambda h: _smooth_grad(h, beta=50.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if res.success or res.fun < best_score:
                h_opt = res.x
                s_opt = _score(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            continue
    
    # Peak-targeted polish: find the lag with max |ac| and try to reduce it
    if time.time() - t0 < DEADLINE:
        ac = _autoconv(best_h)
        max_lag = np.argmax(np.abs(ac))
        # Try local adjustments around the peak
        for _ in range(5):
            if time.time() - t0 > DEADLINE:
                break
            # Perturb near the peak
            h_pert = best_h.copy()
            idx = np.random.randint(0, N)
            h_pert[idx] += np.random.randn() * 0.1
            s_pert = _score(h_pert)
            if s_pert < best_score:
                best_score = s_pert
                best_h = h_pert
    
    return best_h
