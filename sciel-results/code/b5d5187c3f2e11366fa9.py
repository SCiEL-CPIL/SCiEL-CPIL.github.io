"""Minimize the normalized max autoconvolution of a nonnegative 600-step function on [-1/4,1/4]."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 600
HALF = N // 2
DT = (0.5) / N

# EVOLVE-BLOCK-START

def _fft_peak(a):
    """Peak of normalized autoconvolution (a * a) / ||a||_2^2."""
    a = np.asarray(a, dtype=float)
    nrm = np.sqrt(np.dot(a, a))
    if nrm <= 0:
        return 0.0
    conv = fftconvolve(a, a, mode="full")
    return float(np.max(conv) / (nrm * nrm))


def _fourier_optimize(deadline):
    """Optimize a low-dim Fourier parametric representation."""
    M = 16
    a0 = np.zeros(M)
    a0[0] = 1.0
    best_a = a0.copy()
    
    idx = np.arange(N)
    m = idx - HALF
    x = m * DT
    cos_terms = np.stack([np.cos(2.0 * np.pi * k * x) for k in range(M)], axis=1)
    
    def obj(a):
        y = np.maximum(cos_terms @ a, 0.0)
        nrm = np.dot(y, y)
        if nrm <= 0:
            return 1e30
        conv = fftconvolve(y, y, mode="full")
        return float(np.max(conv) / nrm)
    
    best_val = obj(a0)
    for _ in range(3):
        if time.time() > deadline - 25:
            break
        try:
            res = minimize(obj, best_a, method="Nelder-Mead",
                           options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_a = res.x.copy()
        except Exception:
            break
    
    y = np.maximum(cos_terms @ best_a, 0.0)
    return y


def _gauss_optimize(deadline):
    """Optimize a Gaussian mixture representation."""
    n_gauss = 5
    # Parameters: [centers, widths, amplitudes]
    p0 = np.zeros(3 * n_gauss)
    for i in range(n_gauss):
        p0[3*i] = (i - n_gauss//2) * 0.02  # centers
        p0[3*i+1] = 0.01  # widths
        p0[3*i+2] = 1.0   # amplitudes
    
    x_grid = (np.arange(N) - HALF) * DT
    
    def gauss_eval(p):
        y = np.zeros(N)
        for i in range(n_gauss):
            c = p[3*i]
            w = p[3*i+1]
            a = p[3*i+2]
            if w < 1e-6:
                w = 1e-6
            y += a * np.exp(-0.5 * ((x_grid - c) / w) ** 2)
        return np.maximum(y, 0.0)
    
    def obj(p):
        y = gauss_eval(p)
        nrm = np.dot(y, y)
        if nrm <= 0:
            return 1e30
        conv = fftconvolve(y, y, mode="full")
        return float(np.max(conv) / nrm)
    
    best_p = p0.copy()
    best_val = obj(p0)
    
    for _ in range(4):
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(obj, best_p, method="Nelder-Mead",
                           options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            break
    
    y = gauss_eval(best_p)
    return y


def _direct_lbfgs(y, deadline):
    """Direct L-BFGS-B optimization on 600 variables with analytical gradient."""
    a = y.copy()
    a = np.maximum(a, 0.0)
    
    # Precompute for gradient
    # Autoconvolution C = a * a
    # Score = max(C) / (a.a)
    # Let S = a.a, C_max = max(C)
    # d(score)/da_j = (dC_max/da_j * S - C_max * dS/da_j) / S^2
    # dS/da_j = 2*a_j
    # dC_max/da_j = (a * a)_k* where k* is argmax of C
    # (a*a)_k = sum_i a_i a_{i+k}
    # d(a*a)_k/da_j = a_{k-j} + a_{j-k} (with boundary handling)
    
    def obj_grad(a):
        nrm2 = np.dot(a, a)
        if nrm2 <= 0:
            return 1e30, np.zeros_like(a)
        
        conv = fftconvolve(a, a, mode="full")
        k_star = np.argmax(conv)
        C_max = conv[k_star]
        val = C_max / nrm2
        
        # Gradient of C_max w.r.t. a_j
        # C_k = sum_i a_i a_{i+k}
        # dC_k/da_j: j appears in terms where i=j (a_j a_{j+k}) and i+k=j -> i=j-k (a_{j-k} a_j)
        # So dC_k/da_j = a_{j+k} + a_{j-k}
        # But we need to be careful with indices for full convolution
        # conv index k corresponds to lag k - (N-1)
        # Actually, fftconvolve full mode: index 0 is lag -(N-1), index N-1 is lag 0, etc.
        # Let's just compute the gradient numerically for simplicity and correctness, 
        # or use the formula carefully.
        
        # For safety and speed, let's use a semi-analytic approach
        # dC_k/da_j = a_{k-j} + a_{j-k} where indices are relative to the start
        # In full convolution of length 2N-1:
        # conv[m] = sum_i a_i a_{i+m-(N-1)}
        # This is getting complicated. Let's stick to finite differences for the gradient
        # but use a larger step for efficiency, or just use L-BFGS-B without gradient
        # and let it estimate. Given 600 vars, finite difference is 1200 evals per step.
        # L-BFGS-B might do ~100 steps -> 120k evals. Each eval is O(N log N).
        # 120k * 600 * log(600) ~ 120k * 6000 ~ 720M ops. Might be too slow.
        
        # Let's try to derive the gradient properly.
        # a is length N.
        # conv = fftconvolve(a, a, mode='full')
        # conv[k] = sum_{i} a[i] a[i + k - (N-1)]
        # Let m = k - (N-1) be the lag.
        # conv[k] = sum_i a[i] a[i+m]
        # We want d(conv[k_star])/da[j]
        # Term 1: i = j -> a[j] a[j+m]
        # Term 2: i+m = j -> i = j-m -> a[j-m] a[j]
        # So d(conv[k_star])/da[j] = a[j+m] + a[j-m]
        # where m = k_star - (N-1)
        
        m = k_star - (N - 1)
        grad_C = np.zeros(N)
        for j in range(N):
            idx1 = j + m
            idx2 = j - m
            v1 = a[idx1] if 0 <= idx1 < N else 0.0
            v2 = a[idx2] if 0 <= idx2 < N else 0.0
            grad_C[j] = v1 + v2
        
        grad_S = 2.0 * a
        grad_val = (grad_C * nrm2 - C_max * grad_S) / (nrm2 * nrm2)
        
        return val, grad_val
    
    try:
        res = minimize(obj_grad, a, jac=True, method='L-BFGS-B',
                       bounds=[(0, None)] * N,
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
        a_new = np.maximum(res.x, 0.0)
        s_new = _fft_peak(a_new)
        s_old = _fft_peak(a)
        if s_new < s_old:
            return a_new, s_new
    except Exception:
        pass
    
    return a, _fft_peak(a)


def _random_polish(y, deadline, n_restarts=5):
    """Try random perturbations and local polish."""
    best = y.copy()
    best_score = _fft_peak(best)
    
    for _ in range(n_restarts):
        if time.time() > deadline - 15:
            break
        # Random perturbation
        noise = np.random.randn(N) * 0.05 * np.linalg.norm(best)
        cand = np.maximum(best + noise, 0.0)
        
        # Quick local polish with a few L-BFGS steps
        try:
            def obj(a):
                nrm2 = np.dot(a, a)
                if nrm2 <= 0:
                    return 1e30
                conv = fftconvolve(a, a, mode="full")
                return np.max(conv) / nrm2
            
            res = minimize(obj, cand, method='L-BFGS-B',
                           bounds=[(0, None)] * N,
                           options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-6})
            cand = np.maximum(res.x, 0.0)
            s = _fft_peak(cand)
            if s < best_score:
                best = cand
                best_score = s
        except Exception:
            pass
    
    return best, best_score


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin under 150s kill
    
    best = np.ones(N)
    best_score = _fft_peak(best)
    
    # 1) Fourier parametric initialization
    try:
        if time.time() < deadline - 40:
            y = _fourier_optimize(deadline)
            s = _fft_peak(y)
            if s < best_score:
                best = y
                best_score = s
    except Exception:
        pass
    
    # 2) Gaussian mixture initialization
    try:
        if time.time() < deadline - 35:
            y = _gauss_optimize(deadline)
            s = _fft_peak(y)
            if s < best_score:
                best = y
                best_score = s
    except Exception:
        pass
    
    # 3) Direct L-BFGS polish
    try:
        if time.time() < deadline - 20:
            best, best_score = _direct_lbfgs(best, deadline)
    except Exception:
        pass
    
    # 4) Random restarts and polish
    try:
        if time.time() < deadline - 10:
            best, best_score = _random_polish(best, deadline)
    except Exception:
        pass
    
    best = np.maximum(best, 0.0)
    return list(best)

# EVOLVE-BLOCK-END
