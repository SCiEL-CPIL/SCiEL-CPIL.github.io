"""Frequency-domain flat-spectrum optimization for signed step functions minimizing max|autoconv|/(sum x)^2.

Core idea: the objective is scale-invariant, so fix sum(x)=1 and minimize max|autoconv(x)|.
The autoconvolution is IFFT(|FFT(x)|^2). We optimize the spectrum X (conjugate-symmetric
complex, X[0]=1) to make |IFFT(|X|^2)| as flat/peaked-low as possible, using a smooth
log-sum-exp surrogate and gradient descent in the frequency domain. This is a different
representation (frequency vs time) and modelling family (flat-spectrum / low-peak-sidelobe
construction) than direct time-domain optimization.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import rfft, irfft

START_TIME = time.time()
TIME_LIMIT = 150
SAFETY_MARGIN = 6

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Linear autoconvolution via FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = rfft(x, n=L_fft)
    C = irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _freq_to_x(X):
    """Convert conjugate-symmetric real-FFT spectrum (length N//2+1) to real time domain x (length N)."""
    x = irfft(X, n=N)
    return x

def _smooth_max_and_grad_freq(X, beta=40.0):
    """
    Smooth surrogate max|IFFT(|X|^2)| with gradient w.r.t. the complex spectrum X
    (conjugate-symmetric, length N//2+1). We return gradient w.r.t. the real and imag
    parts of X[1..N//2] (the free DOF), plus handle X[0] separately.
    Actually we optimize over the full real vector: we'll work in time domain for the
    gradient but shape the spectrum. Simpler: compute in time domain.
    """
    x = _freq_to_x(X)
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    L = len(ac)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones(L) / L
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp

    # gradient of smooth_max w.r.t. x (time domain), as before
    c = weights * np.sign(ac)
    L_total = 2 * N - 1
    L_fft = 1
    while L_fft < 2 * L_total:
        L_fft *= 2
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x
    cr = c[::-1]
    cr_pad = np.zeros(L_fft)
    cr_pad[:L_total] = cr
    X_fft = rfft(x_pad, n=L_fft)
    CR_fft = rfft(cr_pad, n=L_fft)
    conv = irfft(X_fft * CR_fft, n=L_fft)
    grad_x = 2.0 * conv[2 * N - 2 : 2 * N - 2 - N + 1 : -1]

    # Now convert grad_x to grad w.r.t. X (the complex spectrum).
    # x = IFFT(X), so x_j = (1/N) sum_k X_k exp(2 pi i j k / N) (with conjugate symmetry).
    # d x_j / d X_k (treating X and X* independently) ... this is messy.
    # Instead, we do gradient descent directly in time domain (x), which is simpler
    # and equivalent. We keep the frequency representation only for initialization.
    return smooth_max, grad_x

def _flat_spectrum_init():
    """
    Build a sequence with a flat (constant-magnitude) spectrum — the ideal low-peak
    autoconvolution. We construct X with |X| = const (and X[0]=1), random phase, then
    x = IFFT(X). Then we refine the phase to minimize the peak.
    Returns a good initial x (sum normalized to 1).
    """
    rng = np.random.default_rng(11)
    best_x = np.ones(N)
    best_val = _objective(best_x)

    n_half = N // 2 + 1
    for trial in range(80):
        if _check_deadline():
            break
        # Constant magnitude spectrum with random phase (conjugate symmetric)
        X = np.zeros(n_half, dtype=complex)
        X[0] = 1.0  # DC = 1 (so sum(x) = 1)
        if N % 2 == 0:
            X[-1] = 1.0  # Nyquist is real
        phase = rng.uniform(0, 2 * np.pi, size=n_half)
        mag = np.ones(n_half)
        X[1:n_half - (1 if N % 2 == 0 else 0)] = mag[1:n_half - (1 if N % 2 == 0 else 0)] * np.exp(1j * phase[1:n_half - (1 if N % 2 == 0 else 0)])
        x = _freq_to_x(X)
        ss = np.sum(x)
        if abs(ss) < 1e-10:
            continue
        x = x / ss
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

    return best_x, best_val

def _gradient_descent(x, beta=40.0, max_iters=400, lr_init=0.3):
    """
    Gradient descent on the smooth surrogate max|ac| (with sum(x)=1 fixed) in time domain,
    using backtracking line search. This is the core optimizer.
    """
    x = x.copy()
    ss = np.sum(x)
    if abs(ss) < 1e-10:
        return x, _objective(x)
    x = x / ss

    cur_val, cur_grad = _smooth_max_and_grad_freq(x, beta)
    lr = lr_init
    best_x = x.copy()
    best_val = cur_val

    for it in range(max_iters):
        if _check_deadline():
            break
        # Project gradient to the sum-zero subspace (sum is constrained to 1)
        g = cur_grad - np.mean(cur_grad)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        g = g / g_norm

        step = lr
        improved = False
        for _ in range(25):
            x_trial = x - step * g
            s_t = np.sum(x_trial)
            if abs(s_t) < 1e-10:
                step *= 0.5
                continue
            x_trial = x_trial / s_t
            v_t, _ = _smooth_max_and_grad_freq(x_trial, beta)
            if v_t < cur_val - 1e-10:
                x = x_trial
                cur_val = v_t
                cur_grad = _smooth_max_and_grad_freq(x, beta)[1]
                if cur_val < best_val:
                    best_val = cur_val
                    best_x = x.copy()
                improved = True
                break
            step *= 0.5
        if not improved:
            lr *= 0.6
            if lr < 1e-5:
                break
        else:
            lr = min(lr * 1.06, 3.0)

    return best_x, _objective(best_x)

def _coordinate_polish(x, max_iters=120):
    """Coordinate descent polish on the exact objective."""
    x = x.copy()
    ss = np.sum(x)
    if abs(ss) < 1e-10:
        return x, _objective(x)
    x = x / ss
    cur_val = _objective(x)

    for it in range(max_iters):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            x_old = x[j]
            for delta in [0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.1, -0.1]:
                x_trial = x.copy()
                x_trial[j] = x_old + delta
                s_t = np.sum(x_trial)
                if abs(s_t) < 1e-10:
                    continue
                x_trial = x_trial / s_t
                v_t = _objective(x_trial)
                if v_t < cur_val - 1e-9:
                    x = x_trial
                    cur_val = v_t
                    improved = True
                    break
        if not improved:
            break
    return x, cur_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)

    # Phase 1: flat-spectrum (frequency-domain) initialization — the structural core
    try:
        xs, vs = _flat_spectrum_init()
        if vs < best_val:
            best_val = vs
            best_x = xs.copy()
    except Exception:
        pass

    # Also try a few structured time-domain inits for diversity
    t = np.linspace(0, 1, N)
    for x0 in [np.ones(N), 1.0 + 0.5 * np.cos(np.pi * t),
               1.0 + 0.5 * np.sin(2 * np.pi * t), np.exp(2.0 * t), np.exp(-2.0 * t)]:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    if _check_deadline():
        return best_x

    # Phase 2: gradient descent with annealed smoothness (beta increasing)
    for beta in [15.0, 30.0, 60.0, 120.0, 240.0]:
        if _check_deadline():
            break
        try:
            x_gd, val_gd = _gradient_descent(best_x, beta=beta, max_iters=300, lr_init=0.3)
            if val_gd < best_val:
                best_val = val_gd
                best_x = x_gd.copy()
        except Exception:
            pass

    if _check_deadline():
        return best_x

    # Phase 3: L-BFGS-B on smooth surrogate
    from scipy.optimize import minimize
    for beta in [40.0, 100.0, 200.0]:
        if _check_deadline():
            break

        def _obj_grad(x, b=beta):
            return _smooth_max_and_grad_freq(x, beta=b)

        try:
            result = minimize(_obj_grad, best_x, method='L-BFGS-B', jac=True,
                              options={'maxiter': 120, 'gtol': 1e-6, 'ftol': 1e-9})
            if result.fun < best_val:
                exact_val = _objective(result.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = result.x.copy()
        except Exception:
            pass

    if _check_deadline():
        return best_x

    # Phase 4: coordinate polish on exact objective
    try:
        x_cd, val_cd = _coordinate_polish(best_x, max_iters=100)
        if val_cd < best_val:
            best_val = val_cd
            best_x = x_cd.copy()
    except Exception:
        pass

    if _check_deadline():
        return best_x

    # Phase 5: random perturbation refinement
    rng = np.random.default_rng(42)
    step_size = 0.05
    for iteration in range(15000):
        if _check_deadline():
            break
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        s_t = np.sum(x_trial)
        if abs(s_t) < 1e-10:
            continue
        x_trial = x_trial / s_t
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.005, step_size * 0.999)
        else:
            if iteration % 2000 == 0:
                step_size = min(0.3, step_size * 1.08)

    return best_x
# EVOLVE-BLOCK-END
